"""Qwen3.8-Flash-Next text model with explicit downstream CPU/GGUF seams."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .gdn import Qwen4ExpGatedDeltaNet
from .gguf_attach import (
    GGUFCpuExpertAttachment,
    append_original_expert_state,
    attach_gguf_cpu_eager_bridge,
    attach_gguf_cpu_expert_bundle,
    detach_gguf_cpu_expert_bundle,
    gguf_cpu_expert_telemetry,
)
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer, PLETableBackend, ZeroTable, build_ple_metadata, commit_ngram_context

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig
    from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle
    from freetoken.moe.gguf_transfer import EagerTransferSeam


ObservationHook = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class _MoeExecutionContext:
    phase: str
    group_size: int
    graph_capture: bool
    cache_size: int = 0
    workspace: object | None = None
    num_token_non_padded: int | None = None


def _validate_eager_moe_context(context: _MoeExecutionContext) -> None:
    if context.phase != "decode":
        raise ValueError(
            f"eager GGUF expert attachment is decode-only; phase={context.phase!r} is unsupported"
        )
    if context.group_size != 1:
        raise ValueError(
            "eager GGUF expert attachment requires one request; "
            f"group_size={context.group_size} is unsupported"
        )
    if context.graph_capture:
        raise ValueError("eager GGUF expert attachment cannot run during graph capture")
    if context.cache_size != 0:
        raise ValueError("eager GGUF expert attachment requires cache_size=0")
    if context.workspace is not None:
        raise ValueError("eager GGUF expert attachment does not accept a workspace")


class _SparseMoE(Qwen4ExpMoE):
    """Upstream Qwen MoE plus observer/context forwarding for the borrowed CPU bridge."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        debug_observer: ObservationHook | None = None,
        *,
        execution_context: _MoeExecutionContext | None = None,
    ) -> torch.Tensor:
        eager = bool(getattr(self.experts, "requires_moe_execution_context", False))
        if execution_context is not None:
            _validate_eager_moe_context(execution_context)
        shared_gate_weight = getattr(self.shared_expert_gate, "weight", None)
        if (
            not eager
            and execution_context is None
            and debug_observer is None
            and shared_gate_weight is not None
        ):
            return super().forward(hidden_states)

        from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        if shared_gate_weight is not None:
            gate = shared_gate_sigmoid(hidden_states, shared_gate_weight.view(-1))
        else:
            # Preserve the model-neutral adapter seam used by the GGUF CPU path.
            # Real Qwen modules take the fused weight path above; lightweight
            # adapters may expose only the ordinary module forward contract.
            gate = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)).view(-1)
        kwargs: dict[str, object] = {}
        if debug_observer is not None:
            kwargs["debug_observer"] = debug_observer
        if execution_context is not None:
            kwargs.update(
                phase=execution_context.phase,
                group_size=execution_context.group_size,
                graph_capture=execution_context.graph_capture,
                workspace=execution_context.workspace,
                num_token_non_padded=execution_context.num_token_non_padded,
            )
        routed = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, **kwargs
        )
        if shared_gate_weight is None:
            return (routed + gate[:, None] * shared).view(num_tokens, hidden_dim)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


class _MappedPLETable(PLETableBackend):
    """Adapter from the downstream CPU/NVMe PLE table to the upstream PLE backend contract."""

    def __init__(self, table, head_dim: int):
        try:
            descriptor = table.descriptor
            rows = int(descriptor.rows)
            elements_per_row = int(descriptor.elements_per_row)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("mapped PLE table must expose a valid descriptor geometry") from error
        if rows <= 0 or head_dim <= 0 or elements_per_row != int(head_dim):
            raise ValueError(
                "mapped PLE table geometry does not match the model: "
                f"rows={rows}, elements_per_row={elements_per_row}, head_dim={head_dim}"
            )
        self._table = table
        self.num_rows = rows
        self.head_dim = int(head_dim)
        self.dtype = torch.bfloat16
        self._prefetch_lock = threading.RLock()
        self._prefetch_handle: object | None = None

    @staticmethod
    def _validated_row_ids(row_ids: torch.Tensor) -> np.ndarray:
        if not isinstance(row_ids, torch.Tensor) or row_ids.ndim != 2:
            raise ValueError(
                "PLE row ids must have shape [tokens, heads], "
                f"got {getattr(row_ids, 'shape', None)}"
            )
        if row_ids.dtype == torch.bool or row_ids.is_floating_point() or row_ids.is_complex():
            raise TypeError(f"PLE row ids must be integer, got {row_ids.dtype}")
        return row_ids.detach().cpu().numpy()

    def _wait_for_prefetch_locked(self) -> None:
        handle, self._prefetch_handle = self._prefetch_handle, None
        if handle is None:
            return
        result = getattr(handle, "result", None)
        if not callable(result):
            raise TypeError("mapped PLE prefetch handle must expose result()")
        result()

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        ids = self._validated_row_ids(row_ids)
        with self._prefetch_lock:
            self._wait_for_prefetch_locked()
            values = np.asarray(self._table.lookup(ids)).astype(np.float32, copy=False)
        expected = (*ids.shape, self.head_dim)
        if values.shape != expected:
            raise ValueError(
                "mapped PLE table returned invalid row geometry: "
                f"got {values.shape}, expected {expected}"
            )
        result = torch.as_tensor(values, dtype=torch.bfloat16, device=row_ids.device)
        result = result.reshape(*row_ids.shape[:-1], row_ids.shape[-1] * self.head_dim)
        if out is not None:
            if out.shape != result.shape:
                raise ValueError(
                    "mapped PLE output geometry does not match the flattened backend contract: "
                    f"got {tuple(out.shape)}, expected {tuple(result.shape)}"
                )
            out.copy_(result)
            return out
        return result

    def prefetch(self, row_ids: torch.Tensor) -> None:
        ids = self._validated_row_ids(row_ids)
        with self._prefetch_lock:
            # The downstream table permits one active request.  Waiting here also
            # observes a completed prior handle before replacing it, so worker
            # exceptions cannot disappear behind a later model-layer prefetch.
            self._wait_for_prefetch_locked()
            prefetch = getattr(self._table, "prefetch", None)
            if not callable(prefetch):
                return
            handle = prefetch(ids)
            if handle is not None and not callable(getattr(handle, "result", None)):
                raise TypeError("mapped PLE prefetch handle must expose result()")
            self._prefetch_handle = handle

    def telemetry(self) -> dict[str, object]:
        return self._table.telemetry()

    def close(self) -> None:
        failure: BaseException | None = None
        with self._prefetch_lock:
            try:
                self._wait_for_prefetch_locked()
            except BaseException as error:
                failure = error
            try:
                self._table.close()
            except BaseException as error:
                failure = failure or error
        if failure is not None:
            raise failure


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            group = config.linear_attention_group()
            assert group is not None
            self.linear_attn = Qwen4ExpGatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=group.num_key_heads,
                num_v_heads=group.num_value_heads,
                head_k_dim=group.key_head_dim,
                head_v_dim=group.value_head_dim,
                conv_kernel_size=group.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                output_gate=group.output_gate,
                expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = _SparseMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        hidden: torch.Tensor,
        batch: Batch,
        debug_observer: ObservationHook | None = None,
        *,
        execution_context: _MoeExecutionContext | None = None,
    ) -> torch.Tensor:
        if self.ple is not None:
            contribution = self.ple.forward(hidden, batch)
            if debug_observer is not None:
                debug_observer(
                    "ple",
                    {"layer_id": self._layer_id, "contribution": contribution.detach().clone()},
                )
            hidden = hidden + contribution
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input, debug_observer)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        assert inject is not None
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        block_output = self.mlp.forward(
            block_input, debug_observer, execution_context=execution_context
        )
        assert inject is not None
        return self.mlp_hyper_connection.combine(hidden, block_output, inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)
        self._debug_observer: ObservationHook | None = None
        self._gguf_cpu_attachment: GGUFCpuExpertAttachment | None = None
        self._gguf_attachment_lock = threading.RLock()
        self._ple_tables: list[object] = []
        self._host_resources_closed = False
        self._image_token_id = config.image_token_id

    @property
    def ple_layers(self) -> List[PLELayer]:
        return list(self._ple)

    def set_debug_observer(self, observer: ObservationHook | None) -> None:
        self._debug_observer = observer

    def attach_gguf_cpu_expert_bundle(self, bundle: QwenGGUFCpuExpertBundle) -> None:
        attach_gguf_cpu_expert_bundle(self, bundle)

    def attach_gguf_cpu_eager_bridge(
        self,
        bundle: QwenGGUFCpuExpertBundle,
        *,
        transfer: EagerTransferSeam | None = None,
    ) -> None:
        attach_gguf_cpu_eager_bridge(self, bundle, transfer=transfer)

    def detach_gguf_cpu_expert_bundle(self) -> None:
        detach_gguf_cpu_expert_bundle(self)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        with self._gguf_attachment_lock:
            result = super().state_dict(prefix=prefix, result=result)
            return append_original_expert_state(self, result, prefix=prefix)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        with self._gguf_attachment_lock:
            if getattr(self, "_gguf_cpu_attachment", None) is not None:
                raise RuntimeError("detach the GGUF CPU expert attachment before load_state_dict")
            super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def _close_ple_tables(self) -> None:
        tables = getattr(self, "_ple_tables", ())
        self._ple_tables = []
        # Drop layer references before closing owners so a pinned tensor cannot keep an
        # anonymous HostBank mapping exported while its registration is being released.
        for layer in getattr(self, "_ple", ()):
            layer.ple_embedding.attach_table(None)
        for table in tables:
            close = getattr(table, "close", None)
            if close is not None:
                close()

    def close_host_resources(self) -> None:
        """Close model-owned PLE resources; safe to call repeatedly during teardown."""
        with self._gguf_attachment_lock:
            if getattr(self, "_host_resources_closed", False):
                return
            self._host_resources_closed = True
            self._close_ple_tables()

    def _has_eager_gguf_attachment(self) -> bool:
        attachment = getattr(self, "_gguf_cpu_attachment", None)
        return attachment is not None and attachment.mode == "eager"

    def _attach_ple_table(self, table: PLETableBackend) -> None:
        self._close_ple_tables()
        for layer in self._ple:
            layer.ple_embedding.attach_table(table)
        self._ple_tables.append(table)

    def load_host_tables(self, engine_config) -> int:
        """Compatibility entry point for the upstream engine's explicit host-table phase."""
        return self.load_host_weights(
            engine_config.model_path,
            dummy=getattr(engine_config, "use_dummy_weight", False),
            ple_warm_mode=getattr(engine_config, "ple_warm_mode", "cold"),
            ple_artifact_path=getattr(engine_config, "ple_artifact_path", None),
            ple_backend=getattr(engine_config, "ple_backend", "mmap"),
            ple_planner_mode=getattr(engine_config, "ple_planner_mode", "vectorized"),
            ple_planner_direct_threshold=getattr(
                engine_config, "ple_planner_direct_threshold", 8
            ),
        )

    def load_host_weights(
        self,
        model_path: str,
        *,
        dummy: bool = False,
        ple_warm_mode: str = "cold",
        ple_artifact_path: str | None = None,
        ple_backend: str = "mmap",
        ple_planner_mode: str = "vectorized",
        ple_planner_direct_threshold: int = 8,
    ) -> int:
        if getattr(self, "_host_resources_closed", False):
            raise RuntimeError("Qwen host resources are already closed")
        if not self._ple:
            return 0
        args = self._config.qwen4_args
        if dummy:
            from .ple import derive_ngram_hash_constants

            for layer in self._ple:
                if (
                    args.ple_layer_multipliers is not None
                    and args.ple_head_vocab_sizes is not None
                    and args.ple_head_offsets is not None
                ):
                    mult = args.ple_layer_multipliers
                    sizes = args.ple_head_vocab_sizes
                    offsets = args.ple_head_offsets
                else:
                    mult, sizes, offsets = derive_ngram_hash_constants(
                        vocab_size=self._config.vocab_size,
                        ngram_size=args.ngram_size,
                        num_ngram_heads=args.num_ngram_heads,
                        ngram_vocab_size_base=args.ngram_vocab_size_base,
                        ple_layer_index=layer.ple_index,
                )
                layer.ple_embedding.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                layer.ple_embedding.ngram_heads_vocab_sizes.copy_(
                    torch.tensor(sizes, dtype=torch.int64)
                )
                layer.ple_embedding.ngram_heads_offsets.copy_(
                    torch.tensor(offsets, dtype=torch.int64)
                )
            rows = int(
                layer.ple_embedding.ngram_heads_offsets[-1]
                + layer.ple_embedding.ngram_heads_vocab_sizes[-1]
            )
            self._attach_ple_table(ZeroTable(rows, args.ngram_head_dim))
            return 0

        from freetoken.models.gguf.reader import is_gguf_path

        if ple_artifact_path is not None or is_gguf_path(model_path):
            from freetoken.gguf_host import MappedPLETable

            mapped = (
                MappedPLETable.open_from_artifact(
                    ple_artifact_path,
                    warm_mode=ple_warm_mode,
                    backend=ple_backend,
                    planner_mode=ple_planner_mode,
                    planner_direct_threshold=ple_planner_direct_threshold,
                )
                if ple_artifact_path is not None
                else MappedPLETable.open_from_gguf(
                    model_path,
                    warm_mode=ple_warm_mode,
                    backend=ple_backend,
                    planner_mode=ple_planner_mode,
                    planner_direct_threshold=ple_planner_direct_threshold,
                )
            )
            self._attach_ple_table(_MappedPLETable(mapped, args.ngram_head_dim))
            return int(mapped.descriptor.tensor_bytes)

        from .weight import load_ple_table
        from .ple import PinnedUVATable

        table = load_ple_table(model_path, args)
        self._attach_ple_table(PinnedUVATable(table.bank.tensor, float(table.weight_scale)))
        self._ple_tables.append(table)
        return int(table.bank.nbytes)

    def host_weight_telemetry(self) -> dict[int, dict[str, object]]:
        result: dict[int, dict[str, object]] = {}
        for layer in self._ple:
            table = layer.ple_embedding.table
            telemetry = getattr(table, "telemetry", None)
            result[layer.layer_id] = telemetry() if callable(telemetry) else {"source": type(table).__name__}
        return result

    def debug_state(self) -> dict[int, dict[str, object]]:
        return {layer.layer_id: {} for layer in self._ple}

    def gguf_cpu_expert_telemetry(self) -> dict[int, dict[str, object]]:
        return gguf_cpu_expert_telemetry(self)

    @staticmethod
    def _graph_capture_active(batch: object) -> bool:
        value = getattr(batch, "graph_capture", None)
        if value is not None:
            return bool(value)
        if not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError as error:
            raise RuntimeError("cannot determine CUDA graph-capture state for eager GGUF experts") from error

    @classmethod
    def _eager_execution_context(cls, batch: object) -> _MoeExecutionContext:
        reqs = getattr(batch, "reqs", ())
        valid_tokens = None
        if reqs:
            valid_tokens = sum(int(req.extend_len) for req in reqs)
        context = _MoeExecutionContext(
            phase=str(batch.phase),
            group_size=int(batch.size),
            graph_capture=cls._graph_capture_active(batch),
            num_token_non_padded=valid_tokens,
        )
        _validate_eager_moe_context(context)
        return context

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        *,
        eager: bool,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        batch = batch if batch is not None else get_global_ctx().batch
        execution_context = self._eager_execution_context(batch) if eager else None
        hidden = self.embed_tokens.forward(input_ids)
        if getattr(batch, "mm_embeds", None) is not None or (
            self._image_token_id is not None and bool((input_ids == self._image_token_id).any())
        ):
            raise RuntimeError("Qwen3.8 vision inputs are outside FreeToken-Pascal v1; use text-only prompts")
        hidden = hidden.repeat(1, self.hc_count)
        meta = None
        if self._ple:
            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for layer in self._ple:
                layer.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(
                hidden,
                batch,
                self._debug_observer,
                execution_context=execution_context,
            )
        if meta is not None:
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix(hidden)[0]

    def forward(self, input_ids: torch.Tensor, batch: Batch | None = None) -> torch.Tensor:
        # Attachment mode and dispatch share the attach/detach lock so the expert
        # graph cannot change between the decision and execution.
        with self._gguf_attachment_lock:
            eager = self._has_eager_gguf_attachment()
            if batch is None:
                return self._forward_impl(input_ids, eager=eager)
            return self._forward_impl(input_ids, eager=eager, batch=batch)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            if config.tie_word_embeddings:
                raise AssertionError("NVFP4 lm_head assumes untied embeddings")
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        if config.gguf_model_path is not None:
            from .gguf import convert_qwen4_to_gguf

            convert_qwen4_to_gguf(self, config, model_path=config.gguf_model_path)
        self._debug_hook: Callable[[dict[str, object]], None] | None = None
        self._debug_events: dict[str, list[dict[str, object]]] = {}
        super().__init__()

    def set_debug_hook(self, hook: Callable[[dict[str, object]], None] | None) -> None:
        self._debug_hook = hook
        self.model.set_debug_observer(self._record_debug_event if hook is not None else None)

    def attach_gguf_cpu_expert_bundle(self, bundle: QwenGGUFCpuExpertBundle) -> None:
        self.model.attach_gguf_cpu_expert_bundle(bundle)

    def attach_gguf_cpu_eager_bridge(
        self,
        bundle: QwenGGUFCpuExpertBundle,
        *,
        transfer: EagerTransferSeam | None = None,
    ) -> None:
        self.model.attach_gguf_cpu_eager_bridge(bundle, transfer=transfer)

    def detach_gguf_cpu_expert_bundle(self) -> None:
        self.model.detach_gguf_cpu_expert_bundle()

    def _record_debug_event(self, name: str, payload: dict[str, object]) -> None:
        self._debug_events.setdefault(name, []).append(payload)

    def host_weight_telemetry(self) -> dict[int, dict[str, object]]:
        return self.model.host_weight_telemetry()

    def gguf_cpu_expert_telemetry(self) -> dict[int, dict[str, object]]:
        return self.model.gguf_cpu_expert_telemetry()

    @torch.inference_mode()
    def encode_images(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        del pixel_values, image_grid_thw
        raise RuntimeError("Qwen3.8 vision inputs are outside FreeToken-Pascal v1; use text-only prompts")

    def load_host_weights(self, model_path: str, **kwargs) -> int:
        return self.model.load_host_weights(model_path, **kwargs)

    def load_host_tables(self, engine_config) -> int:
        return self.model.load_host_tables(engine_config)

    def close_host_resources(self) -> None:
        self.model.close_host_resources()

    def _record_debug_hook(self, logits: torch.Tensor) -> None:
        if self._debug_hook is not None:
            self._debug_hook(
                {
                    "logits": logits.detach().clone(),
                    "ple_state": self.model.debug_state(),
                    "observations": self._debug_events,
                }
            )

    def forward(self) -> torch.Tensor:
        if self._debug_hook is not None:
            self._debug_events = {}
        batch = get_global_ctx().batch
        logits = self.lm_head.forward(self.model.forward(batch.input_ids, batch))
        self._record_debug_hook(logits)
        return logits


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpModel",
]

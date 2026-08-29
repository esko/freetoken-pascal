from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import safetensors
import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.models.qwen4_exp.gguf_attach import (
    GGUFCpuExpertAttachment,
    append_original_expert_state,
    attach_gguf_cpu_eager_bridge,
    attach_gguf_cpu_expert_bundle,
    detach_gguf_cpu_expert_bundle,
    gguf_cpu_expert_telemetry,
)
from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle
from freetoken.moe.gguf_transfer import EagerTransferSeam
from freetoken.utils import download_hf_weight, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

    from .args import Qwen4ExpArgs


ObservationHook = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class _MoeExecutionContext:
    """Explicit per-forward facts needed by an optional eager expert bridge."""

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


def _debug_batch_metadata(
    batch, device: torch.device, token_count: int
) -> dict[str, object]:
    """Return explicit request/token identity for semantic correctness snapshots."""
    reqs = batch.reqs
    padded_reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else reqs
    lengths = [int(req.extend_len) for req in reqs]
    padded_lengths = [int(req.extend_len) for req in padded_reqs]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return {
        "request_uids": torch.tensor(
            [int(req.uid) for req in reqs], dtype=torch.int64, device=device
        ),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "cached_lengths": torch.tensor(
            [int(req.cached_len) for req in reqs], dtype=torch.int64, device=device
        ),
        "device_lengths": torch.tensor(
            [int(req.device_len) for req in reqs], dtype=torch.int64, device=device
        ),
        "boundary_positions": torch.tensor(
            [max(int(req.device_len) - 1, -1) for req in reqs],
            dtype=torch.int64,
            device=device,
        ),
        "phase": batch.phase,
        "valid_request_count": len(reqs),
        "valid_token_count": sum(lengths),
        "padded_request_count": len(padded_reqs),
        "padded_token_count": sum(padded_lengths),
        "token_count": int(token_count),
    }


class _GroupedRMSNorm(BaseOP):
    def __init__(self, size: int, group_size: int, eps: float):
        if size % group_size:
            raise ValueError(f"RMSNorm size {size} is not divisible by group size {group_size}")
        self.weight = torch.empty(size)
        self.group_size = group_size
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=hidden,
            weight=self.weight,
            bias=None,
            eps=self.eps,
            group_size=self.group_size,
            is_rms_norm=True,
            weight_plus_one=True,
        )


class _GatedRMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, activation: str):
        self.weight = torch.empty(size)
        self.eps = eps
        self.activation = activation

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=hidden,
            weight=self.weight,
            bias=None,
            z=gate,
            eps=self.eps,
            is_rms_norm=True,
            norm_before_gate=True,
            activation=self.activation,
        )


class _GatedResidual(BaseOP):
    def __init__(self, config: ModelConfig, combine: bool = True):
        args: Qwen4ExpArgs = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        hc_size = self.hc_count * self.hidden_size
        self.hc_norm = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.input_mix_weight_down = LinearReplicated(hc_size, args.hc_lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(args.hc_lowrank, hc_size, has_bias=False)
        self.block_inject_weight = (
            LinearReplicated(hc_size, self.hc_count, has_bias=False) if combine else None
        )

    def forward(self, hyper_input: torch.Tensor):
        normalized = self.hc_norm.forward(hyper_input)
        mix = F.silu(self.input_mix_weight_down.forward(normalized) / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up.forward(mix))
        mix = mix.view(-1, self.hc_count, self.hidden_size)
        mixed = (mix * normalized.view(-1, self.hc_count, self.hidden_size)).mean(dim=1)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * torch.sigmoid(self.block_inject_weight.forward(normalized) / self.hc_count)
        return mixed, hyper_input, inject


class _SharedExpert(BaseOP):
    def __init__(self, config: ModelConfig):
        width = config.shared_expert_intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [width, width], has_bias=False
        )
        self.down_proj = LinearRowParallel(width, config.hidden_size, has_bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(hidden)))


class _SparseMoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        weight_format = "fp8_block" if config.expert_quant == "fp8_block" else "bf16"
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=bool(config.norm_topk_prob),
            weight_format=weight_format,
        )
        # Resident MoELayer instances do not need layer_id for execution, but the
        # semantic router observation must identify their model layer as well.
        self.experts.layer_id = layer_id
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        debug_observer: ObservationHook | None = None,
        *,
        execution_context: _MoeExecutionContext | None = None,
    ) -> torch.Tensor:
        eager = bool(getattr(self.experts, "requires_moe_execution_context", False))
        if eager:
            if execution_context is None:
                raise RuntimeError("eager GGUF expert requires an explicit MoE execution context")
            _validate_eager_moe_context(execution_context)
        router_logits = self.gate.forward(hidden)
        shared = self.shared_expert.forward(hidden)
        shared *= torch.sigmoid(self.shared_expert_gate.forward(hidden))
        if eager:
            eager_kwargs = {
                "phase": execution_context.phase,
                "group_size": execution_context.group_size,
                "graph_capture": execution_context.graph_capture,
                "workspace": execution_context.workspace,
                "num_token_non_padded": execution_context.num_token_non_padded,
            }
            if debug_observer is None:
                routed = self.experts.forward(
                    hidden_states=hidden,
                    router_logits=router_logits,
                    **eager_kwargs,
                )
            else:
                routed = self.experts.forward(
                    hidden_states=hidden,
                    router_logits=router_logits,
                    debug_observer=debug_observer,
                    **eager_kwargs,
                )
        elif debug_observer is None:
            routed = self.experts.forward(hidden_states=hidden, router_logits=router_logits)
        else:
            routed = self.experts.forward(
                hidden_states=hidden,
                router_logits=router_logits,
                debug_observer=debug_observer,
            )
        return routed + shared


def _shift_right_ignore_eos(tokens: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    if shift == 0:
        return tokens
    positions = torch.arange(tokens.numel(), dtype=torch.long)
    eos_positions = torch.where(tokens == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat([eos_positions.new_full((1,), -1), previous_eos_inclusive[:-1]])
    segment_start = previous_eos + 1
    source_positions = positions - shift
    shifted = tokens[source_positions.clamp_min(0)]
    valid = (positions - segment_start >= shift) & (source_positions >= 0)
    return torch.where(valid, shifted, tokens.new_full((), eos_token_id))


def build_ngram_ids(
    tokens: torch.Tensor,
    *,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    tokens = tokens.to(dtype=torch.long, device="cpu")
    shifted = [_shift_right_ignore_eos(tokens, shift, eos_token_id) for shift in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        stop = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        sizes = vocab_sizes[start:stop]
        heads = torch.remainder(mixed.unsqueeze(-1), sizes)
        blocks.append(heads + offsets[start:stop])
    return torch.cat(blocks, dim=-1)


def _ple_request_tokens(req, forwarded_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Return the complete host token history visible to this forward.

    The overlap scheduler advances ``device_len`` before it drains the prior
    sampled token to ``req.input_ids``. During decode, that one current token is
    already present in ``batch.input_ids``. Join it to the committed host prefix
    so PLE hashes the same history as a non-overlapped forward.
    """
    host_len = req.input_ids.numel()
    if host_len >= req.device_len:
        return req.input_ids[: req.device_len]
    if host_len != req.cached_len:
        raise RuntimeError(
            "Qwen4-Exp PLE host history has an unexpected gap: "
            f"host={host_len}, cached={req.cached_len}, device={req.device_len}"
        )
    if forwarded_ids is None or forwarded_ids.numel() != req.extend_len:
        actual = 0 if forwarded_ids is None else forwarded_ids.numel()
        raise RuntimeError(
            "Qwen4-Exp PLE needs the current forwarded tokens: "
            f"got {actual}, expected {req.extend_len}"
        )
    return torch.cat((req.input_ids[: req.cached_len], forwarded_ids.to(device="cpu")))


class _HostNGramEmbedding(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: Qwen4ExpArgs = config.qwen4_args
        self.layer_id = layer_id
        self.ngram_size = args.ngram_size
        self.heads_per_ngram = args.heads_per_ngram
        self.eos_token_id = args.eos_token_id
        self.embedding_dim = args.ple_embed_dim
        self.split_ngram_parts = args.split_ngram_parts
        self.ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
        self.head_dim = self.embedding_dim // self.ngram_heads
        self.layer_multipliers = (
            torch.empty(args.ngram_size, dtype=torch.long)
            if args.ple_layer_multipliers is None
            else torch.tensor(args.ple_layer_multipliers, dtype=torch.long)
        )
        self.ngram_heads_vocab_sizes = (
            torch.empty(self.ngram_heads, dtype=torch.long)
            if args.ple_head_vocab_sizes is None
            else torch.tensor(args.ple_head_vocab_sizes, dtype=torch.long)
        )
        self.ngram_heads_offsets = (
            torch.empty(self.ngram_heads, dtype=torch.long)
            if args.ple_head_offsets is None
            else torch.tensor(args.ple_head_offsets, dtype=torch.long)
        )
        self._handles = []
        self._shards: list[torch.Tensor] = []
        self._shard_ends = torch.empty(0, dtype=torch.long)
        self._scale = torch.tensor(1.0, dtype=torch.bfloat16)
        self._host_constants: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._dummy = False
        self._gguf_ple = None

    def telemetry(self) -> dict[str, object]:
        if self._gguf_ple is None:
            return {
                "source": "dummy" if self._dummy else "safetensors",
                "mapped_bytes": 0,
            }
        return {"source": "gguf-mmap", **self._gguf_ple.telemetry()}

    def load_host_weights(
        self,
        model_path: str,
        *,
        dummy: bool = False,
        ple_warm_mode: str = "cold",
    ) -> None:
        if dummy:
            self._dummy = True
            return
        from freetoken.models.gguf.reader import is_gguf_path

        if is_gguf_path(model_path):
            from freetoken.gguf_host import MappedPLETable

            self._gguf_ple = MappedPLETable.open_from_gguf(
                model_path,
                warm_mode=ple_warm_mode,
            )
            self._host_constants = (
                self.layer_multipliers.cpu(),
                self.ngram_heads_vocab_sizes.cpu(),
                self.ngram_heads_offsets.cpu(),
            )
            try:
                expected_rows = int(
                    self._host_constants[1][-1] + self._host_constants[2][-1]
                )
                if self._gguf_ple.descriptor.rows < expected_rows:
                    raise RuntimeError(
                        f"PLE table has {self._gguf_ple.descriptor.rows} rows, "
                        f"needs {expected_rows}"
                    )
                if self._gguf_ple.descriptor.elements_per_row != self.head_dim:
                    raise RuntimeError(
                        "PLE table row width "
                        f"{self._gguf_ple.descriptor.elements_per_row} != {self.head_dim}"
                    )
            except BaseException:
                self._gguf_ple.close()
                self._gguf_ple = None
                raise
            return
        folder = download_hf_weight(model_path)
        index_path = os.path.join(folder, "model.safetensors.index.json")
        with open(index_path) as index_file:
            weight_map = json.load(index_file)["weight_map"]
        prefix = f"model.language_model.layers.{self.layer_id}.ple.ple_embedding.ngram_embedding"
        shard_count = len([key for key in weight_map if key.startswith(prefix + ".shard_")])
        if shard_count != self.split_ngram_parts:
            raise RuntimeError(
                f"Qwen4-Exp PLE has {shard_count} shards, expected {self.split_ngram_parts}"
            )
        shard_keys = [f"{prefix}.shard_{shard_id}.weight" for shard_id in range(shard_count)]
        if not shard_keys or any(key not in weight_map for key in shard_keys):
            raise RuntimeError(f"Incomplete Qwen4-Exp PLE shards under {prefix}")

        handles = {}
        shards = []
        scale_key = prefix + ".weight_scale"
        if scale_key not in weight_map:
            raise RuntimeError(f"Qwen4-Exp PLE is missing {scale_key}")
        try:
            shard_shape = None
            for key in shard_keys:
                filename = weight_map[key]
                handle = handles.get(filename)
                if handle is None:
                    handle = safetensors.safe_open(
                        os.path.join(folder, filename), framework="pt", device="cpu"
                    ).__enter__()
                    handles[filename] = handle
                shard = handle.get_tensor(key)
                if (
                    shard.dtype != torch.float8_e4m3fn
                    or shard.ndim != 2
                    or shard.shape[0] <= 0
                    or shard.shape[1] != self.head_dim
                ):
                    raise RuntimeError(
                        f"Unexpected PLE shard {key}: {shard.dtype} {tuple(shard.shape)}; "
                        f"expected a non-empty [rows, {self.head_dim}] float8 tensor"
                    )
                if shard_shape is None:
                    shard_shape = tuple(shard.shape)
                elif tuple(shard.shape) != shard_shape:
                    raise RuntimeError(
                        f"PLE table shard {key} shape {tuple(shard.shape)} disagrees with "
                        f"the first shard {shard_shape}"
                    )
                shards.append(shard.view(torch.uint8))

            scale_filename = weight_map[scale_key]
            scale_handle = handles.get(scale_filename)
            if scale_handle is None:
                scale_handle = safetensors.safe_open(
                    os.path.join(folder, scale_filename), framework="pt", device="cpu"
                ).__enter__()
                handles[scale_filename] = scale_handle
            scale = scale_handle.get_tensor(scale_key)
            if (
                not scale.is_floating_point()
                or scale.numel() != 1
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
            ):
                raise RuntimeError(
                    f"Qwen4-Exp PLE {scale_key} must be one finite positive "
                    "floating-point value, "
                    f"got {scale.dtype} with shape {tuple(scale.shape)}"
                )

            shard_ends = torch.tensor([shard.shape[0] for shard in shards]).cumsum(0)
            host_constants = (
                self.layer_multipliers.cpu(),
                self.ngram_heads_vocab_sizes.cpu(),
                self.ngram_heads_offsets.cpu(),
            )
            expected_rows = int(host_constants[1][-1] + host_constants[2][-1])
            if int(shard_ends[-1]) < expected_rows:
                raise RuntimeError(
                    f"PLE table has {int(shard_ends[-1])} rows, needs {expected_rows}"
                )
        except BaseException:
            for handle in reversed(tuple(handles.values())):
                handle.__exit__(None, None, None)
            raise

        self._handles = list(handles.values())
        self._shards = shards
        self._shard_ends = shard_ends
        self._scale = scale.reshape(())
        self._host_constants = host_constants

    def _current_ngram_ids(self) -> torch.Tensor:
        if self._host_constants is None:
            raise RuntimeError("Qwen4-Exp PLE host weights are not loaded")
        batch = get_global_ctx().batch
        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        multipliers, vocab_sizes, offsets = self._host_constants
        pieces = []
        forwarded_host = None
        forwarded_offset = 0
        for req in reqs:
            extend_len = req.extend_len
            forwarded = None
            if req.input_ids.numel() < req.device_len:
                if forwarded_host is None:
                    forwarded_host = batch.input_ids.detach().to(device="cpu")
                forwarded = forwarded_host[forwarded_offset : forwarded_offset + extend_len]
            tokens = _ple_request_tokens(req, forwarded)
            all_ids = build_ngram_ids(
                tokens,
                ngram_size=self.ngram_size,
                heads_per_ngram=self.heads_per_ngram,
                eos_token_id=self.eos_token_id,
                multipliers=multipliers,
                vocab_sizes=vocab_sizes,
                offsets=offsets,
            )
            pieces.append(all_ids[req.cached_len : req.device_len])
            forwarded_offset += extend_len
        result = torch.cat(pieces, dim=0)
        if result.shape[0] != batch.input_ids.numel():
            raise RuntimeError(
                f"PLE token count {result.shape[0]} does not match batch {batch.input_ids.numel()}"
            )
        return result

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._dummy:
            token_count = get_global_ctx().batch.input_ids.numel()
            return torch.zeros(token_count, self.embedding_dim, device=device, dtype=dtype)
        ngram_ids = self._current_ngram_ids().reshape(-1)
        if self._gguf_ple is not None:
            embedded = self._gguf_ple.lookup(ngram_ids.detach().cpu().numpy())
            return torch.from_numpy(embedded).reshape(-1, self.embedding_dim).to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
        shard_ids = torch.bucketize(ngram_ids, self._shard_ends, right=True)
        output = torch.empty(
            ngram_ids.numel(),
            self.head_dim,
            dtype=torch.uint8,
            pin_memory=torch.cuda.is_available(),
        )
        starts = torch.cat([self._shard_ends.new_zeros(1), self._shard_ends[:-1]])
        for shard_id in shard_ids.unique().tolist():
            positions = torch.nonzero(shard_ids == shard_id, as_tuple=False).flatten()
            local_ids = ngram_ids.index_select(0, positions) - starts[shard_id]
            rows = self._shards[shard_id].index_select(0, local_ids)
            output.index_copy_(0, positions, rows)
        fp8 = output.to(device=device, non_blocking=True).view(torch.float8_e4m3fn)
        embedded = fp8.to(dtype) * self._scale.to(device=device, dtype=dtype)
        return embedded.view(-1, self.embedding_dim)


class _DepthwiseConv(BaseOP):
    def __init__(self, channels: int, kernel_size: int):
        self.weight = torch.empty(channels, 1, kernel_size)


class _PLELayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args: Qwen4ExpArgs = config.qwen4_args
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_count = args.hc_count
        hc_size = self.hidden_size * self.hc_count
        self.ple_embedding = _HostNGramEmbedding(config, layer_id)
        self.key_proj = LinearReplicated(args.ple_embed_dim, hc_size, has_bias=False)
        self.value_proj = LinearReplicated(args.ple_embed_dim, self.hidden_size, has_bias=False)
        self.norm_key = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.norm_query = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.norm_conv = _GroupedRMSNorm(hc_size, self.hidden_size, config.rms_norm_eps)
        self.conv1d = _DepthwiseConv(hc_size, args.ple_conv_kernel_size)
        self.dilation = args.ngram_size
        self.state_len = (args.ple_conv_kernel_size - 1) * self.dilation
        self._conv_states: dict[int, torch.Tensor] = {}

    def load_host_weights(
        self,
        model_path: str,
        *,
        dummy: bool = False,
        ple_warm_mode: str = "cold",
    ) -> None:
        self.ple_embedding.load_host_weights(
            model_path,
            dummy=dummy,
            ple_warm_mode=ple_warm_mode,
        )

    def semantic_debug_state(self, batch) -> dict[str, object]:
        """Return active-request PLE state without exposing allocator slot identity."""
        reqs = batch.reqs
        states = []
        for req in reqs:
            state = self._conv_states.get(req.table_idx)
            if state is None:
                raise RuntimeError(
                    f"PLE state missing for active request {req.uid} at layer {self.layer_id}"
                )
            states.append(state.detach().clone())
        if states:
            state_tensor = torch.stack(states, dim=0)
        else:
            state_tensor = torch.empty(
                (0, 1, self.conv1d.weight.shape[0], self.state_len),
                dtype=self.conv1d.weight.dtype,
                device=self.conv1d.weight.device,
            )
        return {
            "request_uids": torch.tensor(
                [int(req.uid) for req in reqs], dtype=torch.int64, device=state_tensor.device
            ),
            "cached_lengths": torch.tensor(
                [int(req.cached_len) for req in reqs],
                dtype=torch.int64,
                device=state_tensor.device,
            ),
            "device_lengths": torch.tensor(
                [int(req.device_len) for req in reqs],
                dtype=torch.int64,
                device=state_tensor.device,
            ),
            "boundary_positions": torch.tensor(
                [max(int(req.device_len) - 1, -1) for req in reqs],
                dtype=torch.int64,
                device=state_tensor.device,
            ),
            "state": state_tensor,
        }

    def host_weight_telemetry(self) -> dict[str, object]:
        return self.ple_embedding.telemetry()

    def _short_conv(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        reqs = batch.padded_reqs if batch.is_decode else batch.reqs
        outputs = []
        offset = 0
        weight = self.conv1d.weight
        for req in reqs:
            length = req.extend_len
            current = hidden[offset : offset + length].transpose(0, 1).unsqueeze(0)
            state = self._conv_states.get(req.table_idx)
            if req.cached_len == 0:
                state = current.new_zeros(1, current.shape[1], self.state_len)
            elif state is None:
                raise RuntimeError(
                    "Qwen4-Exp PLE state cannot resume a radix prefix; "
                    "serve with --cache-type naive"
                )
            combined = torch.cat([state, current], dim=-1)
            convolved = F.conv1d(
                combined,
                weight,
                groups=weight.shape[0],
                dilation=self.dilation,
            )
            outputs.append(F.silu(convolved).squeeze(0).transpose(0, 1))
            self._conv_states[req.table_idx] = combined[..., -self.state_len :].detach()
            offset += length
        return torch.cat(outputs, dim=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        embeddings = self.ple_embedding.forward(hidden.device, hidden.dtype)
        key = self.norm_key.forward(self.key_proj.forward(embeddings))
        key = key.view(-1, self.hc_count, self.hidden_size)
        value = self.value_proj.forward(embeddings)
        query = self.norm_query.forward(hidden).view(-1, self.hc_count, self.hidden_size)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(1)).flatten(1)
        normalized = self.norm_conv.forward(gated)
        return gated + self._short_conv(normalized)


class _QSAIndexer(BaseOP):
    """Qwen4-Exp's weight-free four-head compressed-key indexer."""

    def __init__(self, config: ModelConfig, rotary):
        args: Qwen4ExpArgs = config.qwen4_args
        self.num_q_heads = args.indexer_n_heads
        self.num_kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.q_dim = self.num_q_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.index_qk_proj = LinearReplicated(
            config.hidden_size, self.q_dim + self.k_dim, has_bias=False
        )
        self.q_layernorm = GemmaPlusOneRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_layernorm = GemmaPlusOneRMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary = rotary
        if self.rotary.rotary_dim > self.head_dim:
            raise ValueError(
                f"QSA index head {self.head_dim} is smaller than rotary dim "
                f"{self.rotary.rotary_dim}"
            )

    def _apply_rope(self, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        shape = tensor.shape
        flat = tensor.reshape(shape[0], -1).contiguous()
        # The shared RoPE object was built for 256-wide main heads.  Call its
        # kernel with the 128-wide QSA head size instead of using forward(),
        # which would interpret the fused index-query row with the wrong stride.
        dummy_key = torch.zeros(shape[0], self.head_dim, dtype=tensor.dtype, device=tensor.device)
        self.rotary.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=flat,
            key=dummy_key,
            head_size=self.head_dim,
            cos_sin_cache=self.rotary._cos_sin_cache,
            is_neox=self.rotary.is_neox,
        )
        return flat.view(shape)

    def project(self, hidden: torch.Tensor, positions: torch.Tensor):
        qk = self.index_qk_proj.forward(hidden)
        q_raw, k_raw = torch.split(qk, (self.q_dim, self.k_dim), dim=-1)
        q = q_raw.view(-1, self.num_q_heads, self.head_dim).contiguous()
        k = k_raw.view(-1, self.num_kv_heads, self.head_dim).contiguous()
        q = self.q_layernorm.forward(q)
        q = self._apply_rope(q, positions)
        return q, k

    def normalize_compressed_keys(
        self, keys: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        keys = self.k_layernorm.forward(keys.contiguous())
        return self._apply_rope(keys, positions)


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(config, layer_id)
        # Qwen4 stores centered q/k norm weights (effective scale is 1 + w).
        self.q_norm = GemmaPlusOneRMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(config.head_dim, config.rms_norm_eps)
        self.indexer = _QSAIndexer(config, self.rotary)

    @nvtx_annotate("QSA")
    def forward(
        self,
        x: torch.Tensor,
        debug_observer: ObservationHook | None = None,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        rope_positions = ctx.batch.positions
        q, k, v, gate = self._project(x, rope_positions)
        index_q, index_k = self.indexer.project(x, rope_positions)
        if debug_observer is None:
            output = ctx.attn_backend.qsa_forward(
                q, k, v, index_q, index_k, self.indexer, self.layer_id, ctx.batch
            )
        else:
            from freetoken.attention.qsa import QSAAttnBackend

            if isinstance(ctx.attn_backend, QSAAttnBackend):
                output = ctx.attn_backend.qsa_forward(
                    q,
                    k,
                    v,
                    index_q,
                    index_k,
                    self.indexer,
                    self.layer_id,
                    ctx.batch,
                    debug_observer=debug_observer,
                )
            else:
                # Alternate QSA backends retain their existing call signature. They
                # simply do not provide the optional logical-row snapshot.
                output = ctx.attn_backend.qsa_forward(
                    q, k, v, index_q, index_k, self.indexer, self.layer_id, ctx.batch
                )
        return self._combine(output, gate)


class Qwen4ExpDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        dense_config = replace(config, expert_quant="none", attn_quant="none")
        if self._is_linear:
            group = config.linear_attention_group()
            assert group is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=group.num_key_heads,
                num_v_heads=group.num_value_heads,
                head_k_dim=group.key_head_dim,
                head_v_dim=group.value_head_dim,
                conv_kernel_size=group.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant="none",
                attn_quant="none",
            )
            self.linear_attn.norm = _GatedRMSNorm(
                group.value_head_dim,
                config.rms_norm_eps,
                config.qwen4_args.output_gate_type,
            )
        else:
            self.self_attn = Qwen4ExpAttention(dense_config, layer_id)
        self.mlp = _SparseMoE(config, layer_id)
        self.ple = (
            _PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )
        self.attn_hyper_connection = _GatedResidual(config)
        self.mlp_hyper_connection = _GatedResidual(config)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        hidden: torch.Tensor,
        debug_observer: ObservationHook | None = None,
        *,
        execution_context: _MoeExecutionContext | None = None,
    ) -> torch.Tensor:
        if self.ple is not None:
            ple_contribution = self.ple.forward(hidden)
            if debug_observer is not None:
                metadata = _debug_batch_metadata(
                    get_global_ctx().batch,
                    ple_contribution.device,
                    ple_contribution.shape[0],
                )
                valid_tokens = int(metadata["valid_token_count"])
                debug_observer(
                    "ple",
                    {
                        "layer_id": self._layer_id,
                        "contribution": ple_contribution[:valid_tokens].detach().clone(),
                        **metadata,
                    },
                )
            hidden = hidden + ple_contribution
        mixed, residual, weights = self.attn_hyper_connection.forward(hidden)
        mixed = (
            self.linear_attn.forward(mixed, debug_observer)
            if self._is_linear
            else self.self_attn.forward(mixed, debug_observer)
        )
        hidden = residual + (mixed.unsqueeze(1) * weights.unsqueeze(-1)).flatten(1)
        mixed, residual, weights = self.mlp_hyper_connection.forward(hidden)
        if execution_context is None:
            mixed = self.mlp.forward(mixed, debug_observer)
        else:
            mixed = self.mlp.forward(
                mixed,
                debug_observer,
                execution_context=execution_context,
            )
        return residual + (mixed.unsqueeze(1) * weights.unsqueeze(-1)).flatten(1)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self._config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = _GatedResidual(config, combine=False)
        self.hc_count = config.qwen4_args.hc_count
        self._image_token_id = config.image_token_id
        self._debug_observer: ObservationHook | None = None
        self._gguf_cpu_attachment: GGUFCpuExpertAttachment | None = None
        self._gguf_attachment_lock = threading.RLock()

    def set_debug_observer(self, observer: ObservationHook | None) -> None:
        """Enable semantic intermediate snapshots for an opt-in correctness probe."""
        self._debug_observer = observer

    def attach_gguf_cpu_expert_bundle(self, bundle: QwenGGUFCpuExpertBundle) -> None:
        """Attach a borrowed decode-only bundle to every Qwen routed-expert layer."""
        attach_gguf_cpu_expert_bundle(self, bundle)

    def attach_gguf_cpu_eager_bridge(
        self,
        bundle: QwenGGUFCpuExpertBundle,
        *,
        transfer: EagerTransferSeam | None = None,
    ) -> None:
        """Attach explicit blocking device bridges around every CPU expert layer."""
        attach_gguf_cpu_eager_bridge(self, bundle, transfer=transfer)

    def detach_gguf_cpu_expert_bundle(self) -> None:
        """Restore the original routed-expert objects without closing the bundle."""
        detach_gguf_cpu_expert_bundle(self)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Keep runtime expert attachment orthogonal to model weight serialization."""
        # Capture the graph and append resident expert state under the same lock as
        # attach/detach. Otherwise a concurrent swap can produce a mixed snapshot.
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
        # Serialize loading with attach/detach so a concurrent attach cannot become
        # active after this check and mutate the graph while BaseOP walks it.
        with self._gguf_attachment_lock:
            if getattr(self, "_gguf_cpu_attachment", None) is not None:
                raise RuntimeError(
                    "detach the GGUF CPU expert attachment before load_state_dict"
                )
            super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

    def load_host_weights(
        self,
        model_path: str,
        *,
        dummy: bool = False,
        ple_warm_mode: str = "cold",
    ) -> None:
        for layer in self.layers.op_list:
            if layer.ple is not None:
                layer.ple.load_host_weights(
                    model_path,
                    dummy=dummy,
                    ple_warm_mode=ple_warm_mode,
                )

    def debug_state(self) -> dict[int, dict[str, object]]:
        batch = get_global_ctx().batch
        return {
            layer._layer_id: layer.ple.semantic_debug_state(batch)
            for layer in self.layers.op_list
            if layer.ple is not None
        }

    def host_weight_telemetry(self) -> dict[int, dict[str, object]]:
        return {
            layer._layer_id: layer.ple.host_weight_telemetry()
            for layer in self.layers.op_list
            if layer.ple is not None
        }

    def gguf_cpu_expert_telemetry(self) -> dict[int, dict[str, object]]:
        """Return request-scoped telemetry from attached expert adapters or bridges."""
        return gguf_cpu_expert_telemetry(self)

    def _has_eager_gguf_attachment(self) -> bool:
        attachment = getattr(self, "_gguf_cpu_attachment", None)
        return attachment is not None and attachment.mode == "eager"

    @staticmethod
    def _graph_capture_active(batch: object) -> bool:
        batch_value = getattr(batch, "graph_capture", None)
        if batch_value is not None:
            return bool(batch_value)
        try:
            if not torch.cuda.is_available():
                return False
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError as error:
            raise RuntimeError(
                "cannot determine CUDA graph-capture state for eager GGUF experts"
            ) from error

    @classmethod
    def _eager_execution_context(cls, batch: object) -> _MoeExecutionContext:
        reqs = getattr(batch, "reqs", ())
        valid_tokens = None
        if reqs:
            try:
                valid_tokens = sum(int(req.extend_len) for req in reqs)
            except (AttributeError, TypeError, ValueError):
                valid_tokens = None
        context = _MoeExecutionContext(
            phase=str(batch.phase),
            group_size=int(batch.size),
            graph_capture=cls._graph_capture_active(batch),
            cache_size=0,
            workspace=None,
            num_token_non_padded=valid_tokens,
        )
        _validate_eager_moe_context(context)
        return context

    def _forward_impl(self, input_ids: torch.Tensor, *, eager: bool) -> torch.Tensor:
        batch = get_global_ctx().batch
        execution_context = self._eager_execution_context(batch) if eager else None
        # Validate eager-only restrictions before embedding, routing, shared-expert
        # work, or any host/device transfer can begin.
        hidden = self.embed_tokens.forward(input_ids)
        if getattr(batch, "mm_embeds", None) is not None or (
            self._image_token_id is not None and bool((input_ids == self._image_token_id).any())
        ):
            raise RuntimeError(
                "Qwen3.8 vision inputs are outside FreeToken-Pascal v1; use text-only prompts"
            )
        hidden = hidden.repeat(1, self.hc_count)
        if self._debug_observer is None:
            for layer in self.layers.op_list:
                if execution_context is None:
                    hidden = layer.forward(hidden)
                else:
                    hidden = layer.forward(hidden, execution_context=execution_context)
        else:
            for layer in self.layers.op_list:
                if execution_context is None:
                    hidden = layer.forward(hidden, self._debug_observer)
                else:
                    hidden = layer.forward(
                        hidden,
                        self._debug_observer,
                        execution_context=execution_context,
                    )
        return self.hyper_connection_mixer.forward(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # The mode decision must be made while holding the same lock as attach and
        # detach; otherwise a concurrent attachment can change the expert object
        # between the check and the dispatch.
        with self._gguf_attachment_lock:
            if self._has_eager_gguf_attachment():
                return self._forward_impl(input_ids, eager=True)
            return self._forward_impl(input_ids, eager=False)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen4ExpModel(config)
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
        """Install an internal correctness hook; production defaults to disabled."""
        self._debug_hook = hook
        set_observer = getattr(self.model, "set_debug_observer", None)
        if set_observer is not None:
            set_observer(self._record_debug_event if hook is not None else None)

    def attach_gguf_cpu_expert_bundle(self, bundle: QwenGGUFCpuExpertBundle) -> None:
        """Attach a borrowed decode-only GGUF CPU bundle to the inner Qwen model."""
        self.model.attach_gguf_cpu_expert_bundle(bundle)

    def attach_gguf_cpu_eager_bridge(
        self,
        bundle: QwenGGUFCpuExpertBundle,
        *,
        transfer: EagerTransferSeam | None = None,
    ) -> None:
        """Attach explicit blocking device bridges to the inner Qwen model."""
        self.model.attach_gguf_cpu_eager_bridge(bundle, transfer=transfer)

    def detach_gguf_cpu_expert_bundle(self) -> None:
        """Detach the borrowed GGUF CPU bundle without closing its host mappings."""
        self.model.detach_gguf_cpu_expert_bundle()

    def _record_debug_event(self, name: str, payload: dict[str, object]) -> None:
        self._debug_events.setdefault(name, []).append(payload)

    def host_weight_telemetry(self) -> dict[int, dict[str, object]]:
        return self.model.host_weight_telemetry()

    def gguf_cpu_expert_telemetry(self) -> dict[int, dict[str, object]]:
        return self.model.gguf_cpu_expert_telemetry()

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        del pixel_values, image_grid_thw
        raise RuntimeError(
            "Qwen3.8 vision inputs are outside FreeToken-Pascal v1; use text-only prompts"
        )

    def load_host_weights(
        self,
        model_path: str,
        *,
        dummy: bool = False,
        ple_warm_mode: str = "cold",
    ) -> None:
        self.model.load_host_weights(
            model_path,
            dummy=dummy,
            ple_warm_mode=ple_warm_mode,
        )

    def forward(self) -> torch.Tensor:
        if self._debug_hook is not None:
            self._debug_events = {}
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden)
        if self._debug_hook is not None:
            self._debug_hook(
                {
                    "logits": logits.detach().clone(),
                    "ple_state": self.model.debug_state(),
                    "observations": self._debug_events,
                }
            )
        return logits


__all__ = ["Qwen4ExpForCausalLM", "build_ngram_ids"]

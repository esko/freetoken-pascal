from __future__ import annotations

import importlib.util
from collections.abc import Callable

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.kernel.triton.fp8_block_linear import Fp8BlockColMerged
from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged
from freetoken.layers import BaseOP, LinearColParallelMerged
from freetoken.models.quant_linear import make_replicated_quant
from freetoken.utils import init_logger

from .gdn_contract import (
    GdnDispatchDecision,
    GdnDispatchError,
    gdn_mode_from_env,
    parse_gdn_mode,
    resolve_gdn_dispatch,
)

logger = init_logger(__name__)
GdnObserver = Callable[[GdnDispatchDecision], None]
GdnPhaseObserver = Callable[[str, str], None]
GDN_PHASE_NAMES = (
    "projection",
    "convolution",
    "qkv_prepare",
    "gate",
    "recurrence_device",
    "metadata_validation",
    "adapter_host_overhead",
    "norm",
    "output_projection",
)
_warned_reference_decisions: set[GdnDispatchDecision] = set()


def _probe_fla_available() -> bool:
    """Check the eligible in-tree FLA dependency without importing it; H2 parity is pending."""

    try:
        return (
            importlib.util.find_spec("triton") is not None
            and importlib.util.find_spec("freetoken.kernel.fla") is not None
        )
    except Exception:
        return False


def _probe_triton_candidate_available() -> bool:
    """The issue-93 candidate is unavailable until its donor is audited/imported."""

    return False


def _device_capability(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return (0, 0)
    return tuple(int(part) for part in torch.cuda.get_device_capability(device))


_GATE_ACTIVATIONS = ("silu", "swish", "sigmoid")


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNorm(BaseOP):
    """RMSNorm of x followed by an ``activation(z)`` gate (HF Qwen4ExpTextRMSNormGated).

    Uses the fused FLA ``rms_norm_gated`` Triton kernel (norm(x) * act(z) in one
    kernel) instead of the unfused pow/mean/rsqrt/mul/act chain, matching the
    ``RMSNormGated`` -- collapses ~8 elementwise kernels per GDN layer into one.
    Qwen3.8-Flash-Next gates with sigmoid where Qwen3.5 gates with silu."""

    def __init__(self, dim: int, eps: float, activation: str):
        # rms_norm_gated drops the gate entirely (no error) for a name it does not know.
        assert activation in _GATE_ACTIVATIONS, f"unsupported GDN output gate {activation!r}"
        self.weight = torch.empty(dim)
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, z: torch.Tensor, *, use_fla: bool = True) -> torch.Tensor:
        if use_fla:
            from freetoken.kernel.fla import rms_norm_gated

            return rms_norm_gated(
                x=x,
                weight=self.weight,
                bias=None,
                z=z,
                eps=self.eps,
                is_rms_norm=True,
                norm_before_gate=True,
                activation=self.activation,
            )
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight.float()
        activation = torch.sigmoid if self.activation == "sigmoid" else F.silu
        return (x * activation(z.float())).to(x_dtype)


class Qwen4ExpGatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.

    ``output_gate`` is the gate activation name from ``LinearGatedDeltaGroupConfig``
    ("sigmoid" for Qwen3.8-Flash-Next).
    """

    def __init__(
        self,
        hidden_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        rms_norm_eps,
        layer_id,
        output_gate: str = "sigmoid",
        expert_quant: str = "none",
        attn_quant: str = "none",
        gdn_mode: str | None = None,
        gdn_observer: GdnObserver | None = None,
        gdn_phase_observer: GdnPhaseObserver | None = None,
        gdn_fla_available: bool | None = None,
        gdn_candidate_available: bool | None = None,
        gdn_pascal_available: bool = False,
    ):
        if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads:
            raise ValueError(
                "GatedDeltaNet requires num_v_heads to be a positive multiple of "
                f"num_k_heads, got {num_v_heads} and {num_k_heads}"
            )
        self.layer_id = layer_id
        # Resolve process-wide settings exactly once. A model instance must not switch backend
        # if a caller mutates its environment or package installation between forwards.
        self._gdn_mode = parse_gdn_mode(gdn_mode) if gdn_mode is not None else gdn_mode_from_env()
        if gdn_observer is not None and not callable(gdn_observer):
            raise TypeError("gdn_observer must be callable")
        if gdn_phase_observer is not None and not callable(gdn_phase_observer):
            raise TypeError("gdn_phase_observer must be callable")
        self._gdn_observer = gdn_observer
        # The model factory leaves this optional evidence hook unset, so ordinary forwards do
        # not allocate events or invoke callbacks.
        self._gdn_phase_observer = gdn_phase_observer
        self._gdn_fla_available = (
            _probe_fla_available() if gdn_fla_available is None else gdn_fla_available
        )
        self._gdn_candidate_available = (
            _probe_triton_candidate_available()
            if gdn_candidate_available is None
            else gdn_candidate_available
        )
        if type(gdn_pascal_available) is not bool:
            raise TypeError("gdn_pascal_available must be a bool")
        # This is an injected qualification gate, not a package-presence probe. The model
        # factory leaves it false until H2 P4 parity explicitly registers the backend.
        self._gdn_pascal_available = gdn_pascal_available
        # The FLA chunk/decode kernels read+write recurrent state and per-chunk h as [V, K],
        # while LinearStatePool declares it [K, V]. Equal dimensions make the tensors shape
        # compatible, not semantically equivalent: canonical axis-order parity remains an H2
        # item, and backend switching/checkpoint parity is not claimed by this H0 slice.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # qkv|z carry a weight scale (block-fp8 weight_scale_inv, or per-tensor FP8
        # weight_scale); b|a stay bf16. Both quant modes therefore split the four-way
        # fusion into an fp8 qkvz GEMM + a bf16 ba GEMM (matches the upstream split).
        self._block_fp8 = expert_quant == "fp8_block"
        self._pertensor_fp8 = attn_quant == "fp8_pertensor"
        self._fp8 = self._block_fp8 or self._pertensor_fp8

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._fp8:
            ColMerged = Fp8BlockColMerged if self._block_fp8 else Fp8PerTensorColMerged
            self.in_proj_qkvz = ColMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(hidden_size, self._in_proj_split, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the FLA kernel reads them as fp32) -- matches HF/in-tree FLA, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = _GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation=output_gate)
        # out_proj follows the checkpoint quant: block-fp8 / per-tensor-fp8 / compressed-tensors
        # NVFP4 (W4A16) / bf16. in_proj_* stay bf16 in every mode (above), so a compressed-tensors
        # NVFP4 checkpoint (attn_quant=="nvfp4") only makes out_proj native FP4.
        self.out_proj = make_replicated_quant(
            expert_quant, attn_quant, self.value_dim, hidden_size, has_bias=False
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        observer = getattr(self, "_gdn_phase_observer", None)
        if observer is not None:
            observer("gate", "begin")
        try:
            beta = b.sigmoid()
            g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
            return g, beta
        finally:
            if observer is not None:
                observer("gate", "end")

    def _decode_gate_params(
        self, selected_implementation: str, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Materialize explicit gates only for decode backends that consume them.

        The fused FLA decode kernel consumes the original ``a`` and ``b`` projections and
        computes its gates internally.  Keeping that path out of ``_gate_params`` preserves
        the default operation order when optional benchmark telemetry is attached.
        """

        if selected_implementation not in ("torch-reference", "pascal-fp32"):
            return None
        return self._gate_params(a, b)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(
        self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state
    ) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(
            x,
            self._conv_weight(),
            pool.conv_states[li],
            cu_seqlens,
            cache_indices,
            has_initial_state,
        )
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _reference_conv_prefill(self, conv_in: torch.Tensor, pool, fla) -> torch.Tensor:
        """Device-neutral causal depthwise convolution for the reference GDN path."""
        li = pool.local_index(self.layer_id)
        state_len = pool.conv_states.shape[-1]
        weight = self._conv_weight().unsqueeze(1)
        proof = getattr(fla, "pascal_metadata_proof", None)
        if proof is None:
            starts = fla.cu_seqlens.detach().cpu().tolist()
            slots = fla.cache_indices.detach().cpu().tolist()
            has_initial_state = (
                None
                if fla.has_initial_state is None
                else fla.has_initial_state.detach().cpu().tolist()
            )
        else:
            slots, starts, proof_initial = proof.values_for(
                proof.slot_indices, proof.cu_seqlens, proof.initial_state
            )
            has_initial_state = proof_initial
        result = torch.empty_like(conv_in)
        for index, slot in enumerate(slots):
            start, end = int(starts[index]), int(starts[index + 1])
            sequence = conv_in[start:end].transpose(0, 1)
            previous = pool.conv_states[li, int(slot)]
            if has_initial_state is not None and not bool(has_initial_state[index]):
                previous = torch.zeros_like(previous)
            context = torch.cat((previous, sequence), dim=-1).unsqueeze(0)
            mixed = F.conv1d(context, weight, groups=self.conv_dim).squeeze(0).transpose(0, 1)
            result[start:end] = F.silu(mixed)
            if state_len:
                pool.conv_states[li, int(slot)].copy_(context[0, :, -state_len:])
        return result

    def _reference_conv_decode(
        self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool
    ) -> torch.Tensor:
        """Single-token causal depthwise convolution without sgl_kernel/Triton imports."""
        li = pool.local_index(self.layer_id)
        state_len = pool.conv_states.shape[-1]
        indices = table_idx.to(dtype=torch.long)
        previous = pool.conv_states[li].index_select(0, indices)
        context = torch.cat((previous, conv_in.unsqueeze(-1)), dim=-1)
        mixed = F.conv1d(
            context,
            self._conv_weight().unsqueeze(1),
            groups=self.conv_dim,
        ).squeeze(-1)
        if state_len:
            pool.conv_states[li].index_copy_(0, indices, context[..., -state_len:])
        return F.silu(mixed)

    def _reference_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_source: torch.Tensor,
        fla,
    ) -> torch.Tensor:
        """Run the exact sequential delta rule while preserving indexed pool state.

        This intentionally favors transparent FP32 arithmetic and a small Python loop over
        launch overhead.  It is the correctness/fallback path; the FLA path remains unchanged
        for a qualified modern GPU.
        """
        output_dtype = v.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        g = g.float()
        beta = beta.float()
        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        # HF/FLA apply the head-key-dimension scale after q/k normalization.
        q = q * (self.head_k_dim**-0.5)

        starts = fla.cu_seqlens.detach().cpu().tolist()
        slots = fla.cache_indices.detach().cpu().tolist()
        output = torch.empty_like(v)
        track_boundaries = {}
        if fla.track_boundary_row is not None:
            boundaries = fla.track_boundary_row.detach().cpu().tolist()
            destinations = fla.track_dst.detach().cpu().tolist()
            for track_index, boundary in enumerate(boundaries):
                track_boundaries[int(boundary)] = (int(destinations[track_index]), track_index)
        tracked: list[tuple[int, torch.Tensor]] = []
        state_dtype = state_source.dtype

        for request_index, slot in enumerate(slots):
            start, end = int(starts[request_index]), int(starts[request_index + 1])
            state = state_source[int(slot)].float().clone()
            for row in range(start, end):
                state = state * g[0, row].exp()[:, None, None]
                kv_memory = (state * k[0, row, :, :, None]).sum(dim=-2)
                delta = (v[0, row] - kv_memory) * beta[0, row, :, None]
                state = state + k[0, row, :, :, None] * delta[:, None, :]
                output[0, row] = (state * q[0, row, :, :, None]).sum(dim=-2)
                boundary = row + 1
                if boundary in track_boundaries:
                    destination, _track_index = track_boundaries[boundary]
                    tracked.append((destination, state.clone()))
            state_source[int(slot)].copy_(state.to(state_dtype))

        if tracked:
            destinations = torch.tensor(
                [destination for destination, _state in tracked],
                dtype=torch.long,
                device=state_source.device,
            )
            states = torch.stack([state for _destination, state in tracked]).to(state_dtype)
            state_source.index_copy_(0, destinations, states)
        return output.to(output_dtype)

    def _write_reference_track_snapshot(self, pool, li: int, conv_in, fla) -> None:
        """Copy raw convolution context for the reference path's tracked boundary."""
        if fla.track_dst is None:
            return
        destinations = fla.track_dst.to(dtype=torch.long)
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
        pool.conv_states[li].index_copy_(0, destinations, conv_win.to(pool.conv_states.dtype))

    def _write_track_snapshot(
        self, pool, li: int, conv_in: torch.Tensor, h: torch.Tensor, fla
    ) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a direct shape-compatible copy
        (h is [V,K], the state pool is [K,V]); canonical axis-order parity remains H2 work.
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    @staticmethod
    def _pascal_capture_active(batch) -> bool:
        """Return whether the explicit Pascal path would run under CUDA capture.

        ``pascal_gdn_recurrence`` validates slot metadata through synchronous host reads and
        lazily JIT-loads its module, so it is intentionally eager-only.  A graph flag supplied
        by a caller is authoritative; otherwise query the current CUDA stream when available.
        """

        configured = getattr(batch, "graph_capture", None)
        if configured is not None:
            if type(configured) is not bool:
                raise GdnDispatchError("pascal-fp32 graph_capture metadata must be a bool")
            return configured
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError as error:
            raise GdnDispatchError(
                "pascal-fp32 cannot determine CUDA graph-capture state"
            ) from error

    def _ensure_pascal_metadata_proof(
        self, fla, device: torch.device, *, pool=None, phase: str | None = None
    ):
        """Lazily issue one private Pascal proof for all layers sharing this metadata."""

        owner = getattr(fla, "_pascal_metadata_owner", None)
        proof_phase = phase if phase is not None else getattr(fla, "_pascal_metadata_phase", None)
        pool_tensors = None
        if pool is not None:
            recurrent = getattr(pool, "recurrent_states", None)
            convolution = getattr(pool, "conv_states", None)
            if isinstance(recurrent, torch.Tensor) and isinstance(convolution, torch.Tensor):
                pool_tensors = (recurrent, convolution)
        proof = getattr(fla, "pascal_metadata_proof", None)
        if proof is not None:
            proof.validate_context(
                owner_token=owner,
                phase=proof_phase,
                device=device,
                pool_tensors=pool_tensors,
                expected_slot_values=getattr(fla, "_pascal_host_slot_values", None),
                expected_offset_values=getattr(fla, "_pascal_host_offset_values", None),
                expected_initial_values=getattr(fla, "_pascal_host_initial_values", None),
            )
            return proof
        slots = getattr(fla, "_pascal_host_slot_values", None)
        offsets = getattr(fla, "_pascal_host_offset_values", None)
        if slots is None or offsets is None:
            return None
        from freetoken.kernel.gdn_pascal import _issue_pascal_gdn_metadata_proof

        proof = _issue_pascal_gdn_metadata_proof(
            device,
            slots,
            offsets,
            initial_values=getattr(fla, "_pascal_host_initial_values", None),
            owner_token=owner,
            phase=proof_phase,
            pool_tensors=pool_tensors,
        )
        fla.pascal_metadata_proof = proof
        return proof

    def _validate_pascal_pool(self, pool, device: torch.device) -> None:
        """Validate both state-pool slabs before projection or any in-place update."""

        if pool is None:
            raise GdnDispatchError("pascal-fp32 requires a linear state pool")
        recurrent = getattr(pool, "recurrent_states", None)
        convolution = getattr(pool, "conv_states", None)
        if not isinstance(recurrent, torch.Tensor) or not isinstance(convolution, torch.Tensor):
            raise GdnDispatchError("pascal-fp32 requires recurrent_states and conv_states tensors")
        if recurrent.device != device or convolution.device != device:
            raise GdnDispatchError("pascal-fp32 state pool must be on the hidden-state device")
        if recurrent.dtype != torch.float32:
            raise GdnDispatchError("pascal-fp32 requires an FP32 recurrent state pool")
        if convolution.dtype != self.conv1d.weight.dtype:
            raise GdnDispatchError(
                "pascal-fp32 convolution state dtype must match the convolution weight"
            )
        if recurrent.ndim != 5:
            raise GdnDispatchError(
                "pascal-fp32 recurrent state pool must have shape [layers, slots, heads, K, V]"
            )
        if convolution.ndim != 4:
            raise GdnDispatchError(
                "pascal-fp32 convolution state pool must have shape [layers, slots, channels, K-1]"
            )
        if recurrent.shape[1] <= 0 or convolution.shape[1] != recurrent.shape[1]:
            raise GdnDispatchError("pascal-fp32 state pool has an invalid slot dimension")
        if tuple(recurrent.shape[2:]) != (
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        ):
            raise GdnDispatchError("pascal-fp32 recurrent state pool has incompatible geometry")
        if tuple(convolution.shape[2:]) != (self.conv_dim, self.conv_kernel_size - 1):
            raise GdnDispatchError("pascal-fp32 convolution state pool has incompatible geometry")
        try:
            pool.local_index(self.layer_id)
        except (AttributeError, KeyError, IndexError) as error:
            raise GdnDispatchError(
                f"pascal-fp32 state pool has no local row for layer {self.layer_id}"
            ) from error

    @staticmethod
    def _pascal_fresh_reset_slots(fla, effective_slots, initial_values):
        """Return proof-owned fresh slots; generic ``fresh_state_indices`` is never trusted."""

        if initial_values is None:
            return None
        proof = getattr(fla, "pascal_metadata_proof", None)
        slots = (
            proof.slot_values
            if proof is not None
            else getattr(fla, "_pascal_host_slot_values", None)
        )
        if slots is None:
            slots = [int(value) for value in effective_slots.detach().cpu().tolist()]
        fresh = [int(slot) for slot, initial in zip(slots, initial_values) if not initial]
        if not fresh:
            return None
        return torch.tensor(fresh, dtype=torch.long, device=effective_slots.device)

    def _validate_pascal_metadata(
        self,
        fla,
        *,
        num_slots: int | None = None,
        num_tokens: int | None = None,
        device: torch.device | None = None,
        phase: str | None = None,
        pool=None,
    ):
        """Validate Pascal metadata before projection/convolution/state mutation.

        The returned slot and offset tensors are the proof-owned effective metadata when a
        scheduler-issued proof is present. Generic FLA metadata remains the direct-call
        fallback and is synchronously checked for ABI range and ragged semantics.
        """

        observer = getattr(self, "_gdn_phase_observer", None)
        if observer is not None:
            observer("metadata_validation", "begin")
        try:
            tracking_fields = (
                "track_dst",
                "track_h_row",
                "track_conv_src",
                "track_boundary_row",
            )
            if any(getattr(fla, name, None) is not None for name in tracking_fields):
                raise GdnDispatchError(
                    "pascal-fp32 does not support GDN tracking/checkpoint-boundary metadata"
                )
            if num_slots is None or num_tokens is None:
                return None
            phase = phase if phase is not None else getattr(fla, "_pascal_metadata_phase", None)
            if phase not in ("prefill", "decode"):
                raise GdnDispatchError(f"pascal-fp32 requires a valid phase, got {phase!r}")
            metadata_phase = getattr(fla, "_pascal_metadata_phase", None)
            if metadata_phase is not None and metadata_phase != phase:
                raise GdnDispatchError("pascal-fp32 metadata phase does not match the batch")
            for name in ("cache_indices", "cu_seqlens"):
                value = getattr(fla, name, None)
                if not isinstance(value, torch.Tensor):
                    raise GdnDispatchError(f"pascal-fp32 {name} must be a tensor")
                if value.ndim != 1 or not value.is_contiguous():
                    raise GdnDispatchError(f"pascal-fp32 {name} must be contiguous rank-1")
                if value.dtype not in (torch.int32, torch.int64):
                    raise GdnDispatchError(
                        f"pascal-fp32 {name} must use int32 or int64 metadata"
                    )
                if device is not None and value.device != device:
                    raise GdnDispatchError(f"pascal-fp32 {name} must be on the model device")
            initial_state = getattr(fla, "has_initial_state", None)
            if phase == "decode":
                if initial_state is not None or getattr(fla, "fresh_state_indices", None) is not None:
                    raise GdnDispatchError(
                        "pascal-fp32 decode must not include initial-state or fresh-slot metadata"
                    )
            else:
                if not isinstance(initial_state, torch.Tensor):
                    raise GdnDispatchError("pascal-fp32 prefill requires initial-state metadata")
                if (
                    initial_state.dtype != torch.bool
                    or initial_state.ndim != 1
                    or not initial_state.is_contiguous()
                    or (device is not None and initial_state.device != device)
                ):
                    raise GdnDispatchError(
                        "pascal-fp32 prefill initial-state metadata must be contiguous bool [B]"
                    )
            if pool is not None:
                self._validate_pascal_pool(pool, device or fla.cache_indices.device)
            from freetoken.kernel.gdn_pascal import validate_pascal_gdn_metadata

            proof = getattr(fla, "pascal_metadata_proof", None)
            return validate_pascal_gdn_metadata(
                fla.cache_indices,
                fla.cu_seqlens,
                num_slots=num_slots,
                num_tokens=num_tokens,
                initial_state=fla.has_initial_state,
                metadata_proof=proof,
                phase=phase,
                owner_token=getattr(fla, "_pascal_metadata_owner", None),
                pool_tensors=(pool.recurrent_states, pool.conv_states) if pool is not None else None,
                expected_slot_values=getattr(fla, "_pascal_host_slot_values", None),
                expected_offset_values=getattr(fla, "_pascal_host_offset_values", None),
                expected_initial_values=getattr(fla, "_pascal_host_initial_values", None),
            )
        finally:
            if observer is not None:
                observer("metadata_validation", "end")

    def _pascal_recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_source: torch.Tensor,
        fla,
    ) -> torch.Tensor:
        """Run the explicit Pascal recurrence over the model's leading batch dimension.

        The model projection path uses ``[1, T, H, D]`` for both prefill and decode, while the
        standalone ABI uses flattened ragged ``[T, H, D]`` tensors.  Keep the conversion here so
        the ABI never has to infer a model leading dimension.  All recurrence inputs and the
        pool are FP32; the caller converts the returned core output back to model dtype before
        the existing norm/output projection path.
        """

        if q.ndim != 4 or q.shape[0] != 1:
            raise GdnDispatchError(
                "pascal-fp32 requires a singleton model leading dimension, "
                f"got q shape {tuple(q.shape)}"
            )
        if state_source.dtype != torch.float32:
            raise GdnDispatchError(
                "pascal-fp32 requires an FP32 recurrent state pool; refusing implicit state cast"
            )

        def flat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[0].to(dtype=torch.float32).contiguous()

        # Scheduler metadata is commonly int64, while the standalone CUDA ABI is explicitly
        # int32. Validate before conversion so malformed external metadata cannot wrap into a
        # valid-looking slot or offset.
        def int32_metadata(name: str, tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dtype not in (torch.int32, torch.int64):
                raise GdnDispatchError(
                    f"pascal-fp32 {name} must use int32 or int64 metadata, got {tensor.dtype}"
                )
            if tensor.numel():
                minimum, maximum = torch.aminmax(tensor)
                limit = torch.iinfo(torch.int32)
                if int(minimum.item()) < limit.min or int(maximum.item()) > limit.max:
                    raise GdnDispatchError(f"pascal-fp32 {name} exceeds the int32 ABI range")
            return tensor.to(dtype=torch.int32).contiguous()

        observer = getattr(self, "_gdn_phase_observer", None)
        if observer is not None:
            observer("metadata_validation", "begin")
        try:
            metadata_proof = getattr(fla, "pascal_metadata_proof", None)
            if metadata_proof is None:
                cache_indices = int32_metadata("cache_indices", fla.cache_indices)
                cu_seqlens = int32_metadata("cu_seqlens", fla.cu_seqlens)
            else:
                # A scheduler-issued proof owns dedicated int32 tensors built from its host
                # tuples. The kernel validates their versioned identity without another
                # synchronous device-to-host copy; generic FLA tensors remain untouched.
                cache_indices = metadata_proof.slot_indices
                cu_seqlens = metadata_proof.cu_seqlens
        finally:
            if observer is not None:
                observer("metadata_validation", "end")

        from freetoken.kernel.gdn_pascal import pascal_gdn_recurrence

        if observer is not None:
            observer("adapter_host_overhead", "begin")
            observer("recurrence_device", "begin")
        try:
            return pascal_gdn_recurrence(
                flat(q),
                flat(k),
                flat(v),
                flat(g),
                flat(beta),
                state_source,
                cache_indices,
                cu_seqlens,
                metadata_proof=metadata_proof,
            ).unsqueeze(0)
        finally:
            if observer is not None:
                observer("recurrence_device", "end")
                observer("adapter_host_overhead", "end")

    def forward(
        self,
        hidden_states: torch.Tensor,
        debug_observer: Callable[[str, dict[str, object]], None] | None = None,
    ) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        requested_mode = self._gdn_mode
        fla_available = self._gdn_fla_available
        candidate_available = self._gdn_candidate_available
        decision = resolve_gdn_dispatch(
            requested_mode=requested_mode,
            capability=_device_capability(hidden_states.device),
            dtype=str(dtype),
            fla_available=fla_available,
            triton_candidate_available=candidate_available,
            pascal_fp32_available=self._gdn_pascal_available,
        )
        if self._gdn_observer is not None:
            self._gdn_observer(decision)
        if debug_observer is not None:
            debug_observer("gdn_backend", decision.as_dict())
        if decision.fallback_reason and decision not in _warned_reference_decisions:
            _warned_reference_decisions.add(decision)
            logger.warning_rank0(
                "Qwen4 GDN: "
                f"{decision.fallback_reason} for capability=sm_{decision.capability[0]}"
                f"{decision.capability[1]}, dtype={decision.dtype}; "
                "using torch-reference fallback."
            )
        if decision.selected_implementation == "triton-candidate":
            # The issue-93 donor candidate is deliberately not wired until H1/H2 audit and
            # parity evidence are complete.  An affirmative injected probe may exercise the
            # contract, but it must never silently run the in-tree FLA implementation.
            raise GdnDispatchError(
                "triton-candidate selected but no Qwen4 GDN candidate implementation is registered"
            )
        if decision.selected_implementation == "pascal-fp32":
            # The path is explicit-only and eager-only.  It deliberately reuses the reference
            # projection/conv/gate/norm/output stages and only substitutes the recurrence kernel.
            # Tracking/checkpoint snapshots require the FLA state-return contract and cannot be
            # silently dropped by this standalone adapter.
            if self._pascal_capture_active(batch):
                raise GdnDispatchError("pascal-fp32 is not supported during CUDA graph capture")

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla
        if decision.selected_implementation == "pascal-fp32":
            self._ensure_pascal_metadata_proof(
                fla,
                hidden_states.device,
                pool=pool,
                phase=batch.phase,
            )
            pascal_effective_metadata = self._validate_pascal_metadata(
                fla,
                num_slots=(
                    int(pool.recurrent_states.shape[1])
                    if isinstance(getattr(pool, "recurrent_states", None), torch.Tensor)
                    else 0
                ),
                num_tokens=int(total),
                device=hidden_states.device,
                phase=batch.phase,
                pool=pool,
            )
            pascal_reset_slots = (
                self._pascal_fresh_reset_slots(
                    fla,
                    pascal_effective_metadata[0],
                    pascal_effective_metadata[2],
                )
                if batch.is_prefill
                else None
            )
        else:
            pascal_effective_metadata = None
            pascal_reset_slots = None

        observer = getattr(self, "_gdn_phase_observer", None)
        if observer is not None:
            observer("projection", "begin")
        try:
            if self._fp8:
                qkvz = self.in_proj_qkvz.forward(hidden_states)
                conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
                ba = self.in_proj_ba.forward(hidden_states)
                b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
            else:
                proj = self.in_proj.forward(hidden_states)
                conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
            z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        finally:
            if observer is not None:
                observer("projection", "end")
        li = pool.local_index(self.layer_id)

        if observer is not None:
            observer("convolution", "begin")
        try:
            if batch.is_decode:
                if decision.selected_implementation == "pascal-fp32":
                    assert pascal_effective_metadata is not None
                    effective_slots = pascal_effective_metadata[0]
                    mixed = self._reference_conv_decode(conv_in, effective_slots, pool)
                elif decision.selected_implementation == "torch-reference":
                    mixed = self._reference_conv_decode(conv_in, fla.cache_indices, pool)
                else:
                    # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
                    # per-request state read/write-by-index, all in one kernel (no gather/scatter,
                    # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
                    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

                    mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            else:
                if decision.selected_implementation in ("torch-reference", "pascal-fp32"):
                    mixed = self._reference_conv_prefill(conv_in, pool, fla)
                else:
                    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

                    mixed = self._conv_prefill(
                        conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state
                    )
        finally:
            if observer is not None:
                observer("convolution", "end")

        if batch.is_decode:
            B = mixed.shape[0]
            if observer is not None:
                observer("qkv_prepare", "begin")
            try:
                qf, kf, vf = torch.split(
                    mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
                )
                q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
                k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
                v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            finally:
                if observer is not None:
                    observer("qkv_prepare", "end")
            gate_params = self._decode_gate_params(decision.selected_implementation, a, b)
            observe_recurrence = (
                observer is not None and decision.selected_implementation != "pascal-fp32"
            )
            if observe_recurrence:
                observer("recurrence_device", "begin")
            try:
                if decision.selected_implementation == "torch-reference":
                    assert gate_params is not None
                    g, beta = gate_params
                    core_out = self._reference_recurrent(
                        q,
                        k,
                        v,
                        g.reshape(1, B, self.num_v_heads),
                        beta.float().reshape(1, B, self.num_v_heads),
                        pool.recurrent_states[li],
                        fla,
                    )
                elif decision.selected_implementation == "pascal-fp32":
                    assert gate_params is not None
                    g, beta = gate_params
                    core_out = self._pascal_recurrent(
                        q,
                        k,
                        v,
                        g.reshape(1, B, self.num_v_heads),
                        beta.float().reshape(1, B, self.num_v_heads),
                        pool.recurrent_states[li],
                        fla,
                    )
                else:
                    core_out = gdn_decode_fla(
                        q,
                        k,
                        v,
                        a,
                        b,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        state_source=pool.recurrent_states[li],
                        indices=fla.cache_indices,
                        cu_seqlens=fla.cu_seqlens,
                        scale=self.head_k_dim**-0.5,
                    )
            finally:
                if observe_recurrence:
                    observer("recurrence_device", "end")
        else:
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            if observer is not None:
                observer("qkv_prepare", "begin")
            try:
                qf, kf, vf = torch.split(
                    mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
                )
                q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
                k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
                v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            finally:
                if observer is not None:
                    observer("qkv_prepare", "end")
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if decision.selected_implementation == "pascal-fp32":
                if pascal_reset_slots is not None:
                    pool.recurrent_states[li].index_fill_(0, pascal_reset_slots, 0.0)
            elif fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            observe_recurrence = (
                observer is not None and decision.selected_implementation != "pascal-fp32"
            )
            if observe_recurrence:
                observer("recurrence_device", "begin")
            try:
                if decision.selected_implementation == "torch-reference":
                    core_out = self._reference_recurrent(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        pool.recurrent_states[li],
                        fla,
                    )
                    self._write_reference_track_snapshot(pool, li, conv_in, fla)
                elif decision.selected_implementation == "pascal-fp32":
                    core_out = self._pascal_recurrent(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        pool.recurrent_states[li],
                        fla,
                    )
                else:
                    track = fla.track_dst is not None
                    result = gdn_prefill_chunk_fla(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        state_source=pool.recurrent_states[li],
                        indices=fla.cache_indices,
                        cu_seqlens=fla.cu_seqlens,
                        scale=self.head_k_dim**-0.5,
                        return_h=track,
                    )
                    if track:
                        core_out, h = result
                        self._write_track_snapshot(pool, li, conv_in, h, fla)
                    else:
                        core_out = result
            finally:
                if observe_recurrence:
                    observer("recurrence_device", "end")

        # Pascal recurrence is FP32 by contract; return to the projection/model dtype before
        # the existing reference norm and output projection so downstream residual arithmetic
        # keeps its established dtype and no hidden FP32 path leaks into the model.
        core_out = core_out.to(dtype).reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        if observer is not None:
            observer("norm", "begin")
        try:
            out = self.norm.forward(
                core_out, z, use_fla=decision.selected_implementation == "fla"
            ).reshape(total, -1)
        finally:
            if observer is not None:
                observer("norm", "end")
        if observer is not None:
            observer("output_projection", "begin")
        try:
            return self.out_proj.forward(out)
        finally:
            if observer is not None:
                observer("output_projection", "end")


__all__ = ["GDN_PHASE_NAMES", "GdnPhaseObserver", "Qwen4ExpGatedDeltaNet"]

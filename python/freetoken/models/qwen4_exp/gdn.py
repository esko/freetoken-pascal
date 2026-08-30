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
        gdn_fla_available: bool | None = None,
        gdn_candidate_available: bool | None = None,
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
        self._gdn_observer = gdn_observer
        self._gdn_fla_available = (
            _probe_fla_available() if gdn_fla_available is None else gdn_fla_available
        )
        self._gdn_candidate_available = (
            _probe_triton_candidate_available()
            if gdn_candidate_available is None
            else gdn_candidate_available
        )
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
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

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
        starts = fla.cu_seqlens.detach().cpu().tolist()
        slots = fla.cache_indices.detach().cpu().tolist()
        has_initial_state = (
            None if fla.has_initial_state is None else fla.has_initial_state.detach().cpu().tolist()
        )
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
        return output.to(v.dtype)

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

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._fp8:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            if decision.selected_implementation == "torch-reference":
                mixed = self._reference_conv_decode(conv_in, fla.cache_indices, pool)
            else:
                # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
                # per-request state read/write-by-index, all in one kernel (no gather/scatter,
                # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
                from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

                mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            if decision.selected_implementation == "torch-reference":
                g, beta = self._gate_params(a, b)
                core_out = self._reference_recurrent(
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
        else:
            if decision.selected_implementation == "torch-reference":
                mixed = self._reference_conv_prefill(conv_in, pool, fla)
            else:
                from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

                mixed = self._conv_prefill(
                    conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state
                )
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
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

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(
            core_out, z, use_fla=decision.selected_implementation == "fla"
        ).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["Qwen4ExpGatedDeltaNet"]

"""Explicit JIT adapter for the standalone Pascal FP32 GDN recurrence.

The adapter is deliberately separate from automatic GDN dispatch.  It validates the
FreeToken ragged state-pool contract before loading CUDA code, and callers must opt into the
``pascal-fp32`` backend through :mod:`freetoken.models.qwen4_exp.gdn_contract`.  No H2 parity
or performance qualification is implied by compiling or invoking this module.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

PASCAL_GDN_HEAD_DIMS = (64, 128)
PASCAL_GDN_SOURCE = "python/freetoken/kernel/csrc/jit/gdn_pascal.cu"


class PascalGdnContractError(ValueError):
    """Raised when tensors cannot satisfy the explicit Pascal GDN ABI."""


@dataclass(frozen=True, slots=True)
class PascalGdnMetadataBinding:
    """Identity and layout of one device metadata tensor bound to a host proof."""

    object_id: int
    data_ptr: int
    storage_offset: int
    stride: tuple[int, ...]
    version: int | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    owner_epoch: int


_METADATA_PROOF_SEAL = object()


def _tensor_version(value: torch.Tensor) -> int | None:
    try:
        return int(value._version)
    except RuntimeError:
        # Inference tensors intentionally have no version counter. Their scheduler-owned
        # metadata proof still binds object/storage/layout and its owner epoch.
        return None


def _metadata_binding(value: torch.Tensor, owner_epoch: int) -> PascalGdnMetadataBinding:
    return PascalGdnMetadataBinding(
        object_id=id(value),
        data_ptr=int(value.data_ptr()),
        storage_offset=int(value.storage_offset()),
        stride=tuple(int(part) for part in value.stride()),
        version=_tensor_version(value),
        shape=tuple(int(part) for part in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        owner_epoch=int(owner_epoch),
    )


@dataclass(frozen=True, slots=True)
class PascalGdnMetadataProof:
    """Host-origin metadata values bound to the exact tensors staged for one forward.

    The scheduler creates this proof after staging ``slot_indices`` and ``cu_seqlens``.
    A Pascal launch can use the immutable host values for range and monotonicity checks
    without copying the same device tensors back to the CPU.  The private seal and trusted
    copies make accidental ``dataclasses.replace`` edits fail closed; tensor identity,
    storage, version, shape, dtype, and device are checked on every launch.
    """

    slot_values: tuple[int, ...]
    offset_values: tuple[int, ...]
    slot_binding: PascalGdnMetadataBinding
    offset_binding: PascalGdnMetadataBinding
    _trusted_slot_values: tuple[int, ...] = field(repr=False, compare=False)
    _trusted_offset_values: tuple[int, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)
    initial_values: tuple[bool, ...] | None = None
    initial_binding: PascalGdnMetadataBinding | None = None
    _trusted_initial_values: tuple[bool, ...] | None = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._seal is not _METADATA_PROOF_SEAL:
            raise PascalGdnContractError("Pascal GDN metadata proof is not scheduler-issued")
        if (
            self.slot_values != self._trusted_slot_values
            or self.offset_values != self._trusted_offset_values
            or self.initial_values != self._trusted_initial_values
        ):
            raise PascalGdnContractError("Pascal GDN metadata proof values were modified")

    @classmethod
    def issue(
        cls,
        slot_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        slot_values: tuple[int, ...],
        offset_values: tuple[int, ...],
        *,
        initial_state: torch.Tensor | None = None,
        initial_values: tuple[bool, ...] | None = None,
        owner_epoch: int = 0,
    ) -> "PascalGdnMetadataProof":
        slots = tuple(int(value) for value in slot_values)
        offsets = tuple(int(value) for value in offset_values)
        initial = None if initial_values is None else tuple(bool(value) for value in initial_values)
        if (initial_state is None) != (initial is None):
            raise PascalGdnContractError(
                "Pascal GDN initial-state proof requires tensor and host values together"
            )
        return cls(
            slot_values=slots,
            offset_values=offsets,
            slot_binding=_metadata_binding(slot_indices, owner_epoch),
            offset_binding=_metadata_binding(cu_seqlens, owner_epoch),
            initial_values=initial,
            initial_binding=(
                None if initial_state is None else _metadata_binding(initial_state, owner_epoch)
            ),
            _trusted_slot_values=slots,
            _trusted_offset_values=offsets,
            _trusted_initial_values=initial,
            _seal=_METADATA_PROOF_SEAL,
        )

    def values_for(
        self,
        slot_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[list[int], list[int], list[bool] | None]:
        """Validate binding and return the immutable host values without device reads."""

        if self._seal is not _METADATA_PROOF_SEAL:
            raise PascalGdnContractError("Pascal GDN metadata proof is not scheduler-issued")
        if (
            self.slot_values != self._trusted_slot_values
            or self.offset_values != self._trusted_offset_values
            or self.initial_values != self._trusted_initial_values
        ):
            raise PascalGdnContractError("Pascal GDN metadata proof values were modified")
        if _metadata_binding(slot_indices, self.slot_binding.owner_epoch) != self.slot_binding:
            raise PascalGdnContractError("Pascal GDN slot metadata proof is stale or unbound")
        if _metadata_binding(cu_seqlens, self.offset_binding.owner_epoch) != self.offset_binding:
            raise PascalGdnContractError("Pascal GDN offset metadata proof is stale or unbound")
        if initial_state is not None and self.initial_binding is not None:
            if _metadata_binding(initial_state, self.initial_binding.owner_epoch) != self.initial_binding:
                raise PascalGdnContractError("Pascal GDN initial-state proof is stale or unbound")
        elif initial_state is not None:
            raise PascalGdnContractError("Pascal GDN initial-state proof is unexpectedly present")
        if len(self.slot_values) != int(slot_indices.numel()):
            raise PascalGdnContractError("Pascal GDN slot metadata proof length mismatch")
        if len(self.offset_values) != int(cu_seqlens.numel()):
            raise PascalGdnContractError("Pascal GDN offset metadata proof length mismatch")
        initial = None if self.initial_values is None else list(self.initial_values)
        return list(self.slot_values), list(self.offset_values), initial


@dataclass(frozen=True, slots=True)
class PascalGdnLaunch:
    """Validated launch geometry for one ragged, slot-indexed GDN operation."""

    head_dim: int
    num_tokens: int
    num_requests: int
    num_k_heads: int
    num_v_heads: int
    num_slots: int


def _require_tensor(name: str, value: torch.Tensor) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_float32(name: str, value: torch.Tensor) -> None:
    import torch

    if value.dtype != torch.float32:
        raise PascalGdnContractError(f"{name} must be float32, got {value.dtype}")


def _require_contiguous(name: str, value: torch.Tensor) -> None:
    if not value.is_contiguous():
        raise PascalGdnContractError(f"{name} must be contiguous")


def _cpu_int_values(name: str, value: torch.Tensor) -> list[int]:
    # Validation is intentionally synchronous: an explicit launch must fail before a bad slot
    # can race another request.  The backend is not graph-selected and is not graph-capturable.
    return [int(item) for item in value.detach().to(device="cpu").flatten().tolist()]


def make_pascal_gdn_metadata_proof(
    slot_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    slot_values: tuple[int, ...],
    offset_values: tuple[int, ...],
    *,
    initial_state: torch.Tensor | None = None,
    initial_values: tuple[bool, ...] | None = None,
    owner_epoch: int = 0,
) -> PascalGdnMetadataProof:
    """Issue a proof for scheduler-owned host values and their staged device tensors."""

    return PascalGdnMetadataProof.issue(
        slot_indices,
        cu_seqlens,
        slot_values,
        offset_values,
        initial_state=initial_state,
        initial_values=initial_values,
        owner_epoch=owner_epoch,
    )


def validate_pascal_gdn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor | None = None,
    metadata_proof: PascalGdnMetadataProof | None = None,
) -> PascalGdnLaunch:
    """Validate the shipping standalone Pascal GDN ABI without compiling CUDA.

    Shapes are ``q/k=[T,HK,D]``, ``v=[T,HV,D]``, ``g/beta=[T,HV]``,
    ``state_pool=[S,HV,K,V]``, ``slot_indices=[B]``, ``cu_seqlens=[B+1]`` and
    ``output=[T,HV,D]``.  ``K=V=D`` is required by the current Qwen4 pool.  ``beta`` is
    explicitly pre-sigmoided and ``g`` is a log-decay; neither is transformed by this adapter.
    """

    import torch

    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "state_pool": state_pool,
        "slot_indices": slot_indices,
        "cu_seqlens": cu_seqlens,
    }
    if output is not None:
        tensors["output"] = output
    for name, value in tensors.items():
        _require_tensor(name, value)

    for name in ("q", "k", "v", "g", "beta", "state_pool"):
        _require_float32(name, tensors[name])
    if output is not None:
        _require_float32("output", output)
    if slot_indices.dtype != torch.int32:
        raise PascalGdnContractError(f"slot_indices must be int32, got {slot_indices.dtype}")
    if cu_seqlens.dtype != torch.int32:
        raise PascalGdnContractError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
    for name, value in tensors.items():
        _require_contiguous(name, value)

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise PascalGdnContractError("q, k and v must be rank-3 tensors")
    if g.ndim != 2 or beta.ndim != 2:
        raise PascalGdnContractError("g and beta must be rank-2 tensors")
    if state_pool.ndim != 4:
        raise PascalGdnContractError("state_pool must be rank-4 [slots, heads, K, V]")
    if slot_indices.ndim != 1 or cu_seqlens.ndim != 1:
        raise PascalGdnContractError("slot_indices and cu_seqlens must be rank-1 tensors")

    num_tokens, num_k_heads, head_dim = (int(part) for part in q.shape)
    if num_tokens <= 0:
        raise PascalGdnContractError("Pascal GDN requires at least one token")
    if head_dim not in PASCAL_GDN_HEAD_DIMS:
        raise PascalGdnContractError(
            f"Pascal GDN head dimension must be one of {PASCAL_GDN_HEAD_DIMS}, got {head_dim}"
        )
    if tuple(k.shape) != (num_tokens, num_k_heads, head_dim):
        raise PascalGdnContractError("k must have the same [T, HK, D] shape as q")
    num_v_tokens, num_v_heads, value_dim = (int(part) for part in v.shape)
    if (num_v_tokens, value_dim) != (num_tokens, head_dim):
        raise PascalGdnContractError("v must have shape [T, HV, D] matching q")
    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads:
        raise PascalGdnContractError("num_v_heads must be a positive multiple of num_k_heads")
    if tuple(g.shape) != (num_tokens, num_v_heads):
        raise PascalGdnContractError("g must have shape [T, HV]")
    if tuple(beta.shape) != (num_tokens, num_v_heads):
        raise PascalGdnContractError("beta must have shape [T, HV]")

    num_slots, state_heads, state_k, state_v = (int(part) for part in state_pool.shape)
    if (state_heads, state_k, state_v) != (num_v_heads, head_dim, head_dim):
        raise PascalGdnContractError("state_pool must have shape [S, HV, D, D]")
    if num_slots <= 0:
        raise PascalGdnContractError("state_pool must contain at least one slot")
    num_requests = int(slot_indices.shape[0])
    if num_requests <= 0:
        raise PascalGdnContractError("slot_indices must contain at least one request")
    if int(cu_seqlens.shape[0]) != num_requests + 1:
        raise PascalGdnContractError("cu_seqlens must contain B + 1 entries")
    if output is not None and tuple(output.shape) != (num_tokens, num_v_heads, head_dim):
        raise PascalGdnContractError("output must have shape [T, HV, D]")
    if output is not None and output.device != q.device:
        raise PascalGdnContractError("output must be on the same device as q")

    for name, value in tensors.items():
        if value.device != q.device:
            raise PascalGdnContractError(f"{name} must be on the same device as q")
    if q.device.type != "cuda":
        # Keep the validator useful for H0 tests; the launch function below gives the stronger
        # device error.  This distinction lets hosted tests inspect all shape/failure paths.
        pass

    if metadata_proof is None:
        slots = _cpu_int_values("slot_indices", slot_indices)
        offsets = _cpu_int_values("cu_seqlens", cu_seqlens)
    else:
        if not isinstance(metadata_proof, PascalGdnMetadataProof):
            raise PascalGdnContractError("metadata_proof must be a Pascal GDN metadata proof")
        slots, offsets, _initial_values = metadata_proof.values_for(slot_indices, cu_seqlens)
    if any(slot < 0 or slot >= num_slots for slot in slots):
        raise PascalGdnContractError("slot_indices contains an out-of-range pool slot")
    if len(set(slots)) != len(slots):
        raise PascalGdnContractError("slot_indices must be unique per launch")
    if (
        offsets[0] != 0
        or offsets[-1] != num_tokens
        or any(left > right for left, right in pairwise(offsets))
    ):
        raise PascalGdnContractError("cu_seqlens must be nondecreasing from 0 to T")
    return PascalGdnLaunch(
        head_dim=head_dim,
        num_tokens=num_tokens,
        num_requests=num_requests,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        num_slots=num_slots,
    )


@functools.cache
def _jit_pascal_gdn_module(head_dim: int) -> Module:
    if head_dim not in PASCAL_GDN_HEAD_DIMS:
        raise PascalGdnContractError(f"unsupported Pascal GDN head dimension: {head_dim}")
    args = make_cpp_args(head_dim)
    return load_jit(
        "gdn_pascal_f32",
        *args,
        cuda_files=["gdn_pascal.cu"],
        cuda_wrappers=[("launch", f"PascalGdnKernel<{args}>::run")],
    )


def pascal_gdn_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    slot_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    metadata_proof: PascalGdnMetadataProof | None = None,
) -> torch.Tensor:
    """Launch the explicit standalone Pascal FP32 recurrence and return ``output``.

    This function does not participate in ``auto`` dispatch.  It rejects CPU execution,
    non-FP32 input, stale/duplicate slots and malformed ragged metadata before JIT loading.
    """

    import torch

    launch = validate_pascal_gdn_inputs(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        slot_indices,
        cu_seqlens,
        output,
        metadata_proof,
    )
    if q.device.type != "cuda":
        raise PascalGdnContractError("Pascal GDN requires a CUDA tensor on an explicit launch")
    if output is None:
        output = torch.empty_like(v)
    module = _jit_pascal_gdn_module(launch.head_dim)
    module.launch(q, k, v, g, beta, state_pool, slot_indices, cu_seqlens, output)
    return output


__all__ = [
    "PASCAL_GDN_HEAD_DIMS",
    "PASCAL_GDN_SOURCE",
    "PascalGdnContractError",
    "PascalGdnLaunch",
    "PascalGdnMetadataBinding",
    "PascalGdnMetadataProof",
    "make_pascal_gdn_metadata_proof",
    "pascal_gdn_recurrence",
    "validate_pascal_gdn_inputs",
]

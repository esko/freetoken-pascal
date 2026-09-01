"""Explicit JIT adapter for the standalone Pascal FP32 GDN recurrence.

The adapter is deliberately separate from automatic GDN dispatch.  It validates the
FreeToken ragged state-pool contract before loading CUDA code, and callers must opt into the
``pascal-fp32`` backend through :mod:`freetoken.models.qwen4_exp.gdn_contract`.  No H2 parity
or performance qualification is implied by compiling or invoking this module.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
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
    version: int
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True, slots=True)
class PascalGdnPoolBinding:
    """Identity and layout of a state-pool tensor bound to a metadata proof."""

    object_id: int
    data_ptr: int
    storage_offset: int
    stride: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    device: str


_METADATA_PROOF_SEAL = object()


def _tensor_version(value: torch.Tensor) -> int:
    try:
        return int(value._version)
    except RuntimeError:
        raise PascalGdnContractError(
            "Pascal GDN metadata proof requires versioned tensors; construct it outside inference mode"
        ) from None


def _metadata_binding(value: torch.Tensor) -> PascalGdnMetadataBinding:
    return PascalGdnMetadataBinding(
        object_id=id(value),
        data_ptr=int(value.data_ptr()),
        storage_offset=int(value.storage_offset()),
        stride=tuple(int(part) for part in value.stride()),
        version=_tensor_version(value),
        shape=tuple(int(part) for part in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _pool_binding(value: torch.Tensor) -> PascalGdnPoolBinding:
    return PascalGdnPoolBinding(
        object_id=id(value),
        data_ptr=int(value.data_ptr()),
        storage_offset=int(value.storage_offset()),
        stride=tuple(int(part) for part in value.stride()),
        shape=tuple(int(part) for part in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _int32_values(name: str, values) -> tuple[int, ...]:
    """Normalize host metadata and reject values that cannot cross the CUDA ABI."""

    import torch

    limit = torch.iinfo(torch.int32)
    normalized = []
    for value in values:
        try:
            converted = operator.index(value)
        except TypeError as error:
            raise PascalGdnContractError(f"{name} must contain integers") from error
        if converted < limit.min or converted > limit.max:
            raise PascalGdnContractError(f"{name} exceeds the int32 ABI range")
        normalized.append(int(converted))
    return tuple(normalized)


def _initial_values(values: tuple[bool, ...] | None) -> tuple[bool, ...] | None:
    if values is None:
        return None
    normalized = []
    for value in values:
        if type(value) is not bool:
            raise PascalGdnContractError("Pascal GDN initial-state proof values must be bool")
        normalized.append(value)
    return tuple(normalized)


class PascalGdnMetadataProof:
    """Host-origin metadata values bound to the exact tensors staged for one forward.

    The scheduler supplies host values and the explicit Pascal model lazily creates the
    proof-owned tensors. A Pascal launch can use the immutable host values for range and
    monotonicity checks without copying generic device tensors back to the CPU. The private
    token and opaque object prevent ``dataclasses.replace`` forgery; tensor identity, storage,
    version, shape, dtype, and device are checked on every launch.
    """

    __slots__ = (
        "_slot_indices",
        "_cu_seqlens",
        "_initial_state",
        "_slot_values",
        "_offset_values",
        "_slot_binding",
        "_offset_binding",
        "_trusted_slot_values",
        "_trusted_offset_values",
        "_seal",
        "_initial_values",
        "_initial_binding",
        "_trusted_initial_values",
        "_owner_token",
        "_phase",
        "_device",
        "_pool_bindings",
        "_initialized",
    )

    def __init__(
        self,
        seal: object,
        slot_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: torch.Tensor | None,
        slot_values: tuple[int, ...],
        offset_values: tuple[int, ...],
        initial_values: tuple[bool, ...] | None,
        *,
        owner_token: object | None = None,
        phase: str | None = None,
        pool_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        if seal is not _METADATA_PROOF_SEAL:
            raise PascalGdnContractError("Pascal GDN metadata proof is not scheduler-issued")
        object.__setattr__(self, "_slot_indices", slot_indices)
        object.__setattr__(self, "_cu_seqlens", cu_seqlens)
        object.__setattr__(self, "_initial_state", initial_state)
        object.__setattr__(self, "_slot_values", tuple(slot_values))
        object.__setattr__(self, "_offset_values", tuple(offset_values))
        object.__setattr__(self, "_slot_binding", _metadata_binding(slot_indices))
        object.__setattr__(self, "_offset_binding", _metadata_binding(cu_seqlens))
        object.__setattr__(self, "_trusted_slot_values", tuple(slot_values))
        object.__setattr__(self, "_trusted_offset_values", tuple(offset_values))
        object.__setattr__(self, "_initial_values", initial_values)
        object.__setattr__(
            self,
            "_initial_binding",
            None if initial_state is None else _metadata_binding(initial_state),
        )
        object.__setattr__(self, "_trusted_initial_values", initial_values)
        object.__setattr__(self, "_owner_token", owner_token)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_device", str(slot_indices.device))
        object.__setattr__(
            self,
            "_pool_bindings",
            None if pool_tensors is None else tuple(_pool_binding(tensor) for tensor in pool_tensors),
        )
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError("Pascal GDN metadata proof is immutable")
        object.__setattr__(self, _name, _value)

    @property
    def slot_indices(self) -> torch.Tensor:
        return self._slot_indices

    @property
    def cu_seqlens(self) -> torch.Tensor:
        return self._cu_seqlens

    @property
    def initial_state(self) -> torch.Tensor | None:
        return self._initial_state

    @property
    def slot_values(self) -> tuple[int, ...]:
        return self._slot_values

    @property
    def offset_values(self) -> tuple[int, ...]:
        return self._offset_values

    @property
    def initial_values(self) -> tuple[bool, ...] | None:
        return self._initial_values

    def validate_context(
        self,
        *,
        owner_token: object | None = None,
        phase: str | None = None,
        device: torch.device | None = None,
        pool_tensors: tuple[torch.Tensor, ...] | None = None,
        expected_slot_values: tuple[int, ...] | None = None,
        expected_offset_values: tuple[int, ...] | None = None,
        expected_initial_values: tuple[bool, ...] | None = None,
    ) -> None:
        """Reject a proof attached to a different forward, phase, or state pool."""

        if self._seal is not _METADATA_PROOF_SEAL:
            raise PascalGdnContractError("Pascal GDN metadata proof is not scheduler-issued")
        if owner_token is not None and self._owner_token is not owner_token:
            raise PascalGdnContractError("Pascal GDN metadata proof belongs to another forward")
        if phase is not None and self._phase != phase:
            raise PascalGdnContractError("Pascal GDN metadata proof phase mismatch")
        if device is not None and self._device != str(device):
            raise PascalGdnContractError("Pascal GDN metadata proof device mismatch")
        if pool_tensors is not None:
            actual = tuple(_pool_binding(tensor) for tensor in pool_tensors)
            if self._pool_bindings != actual:
                raise PascalGdnContractError("Pascal GDN metadata proof state pool mismatch")
        if expected_slot_values is not None and self.slot_values != _int32_values(
            "Pascal GDN slot metadata", expected_slot_values
        ):
            raise PascalGdnContractError("Pascal GDN metadata proof slot values mismatch")
        if expected_offset_values is not None and self.offset_values != _int32_values(
            "Pascal GDN offset metadata", expected_offset_values
        ):
            raise PascalGdnContractError("Pascal GDN metadata proof offset values mismatch")
        if expected_initial_values is not None and self.initial_values != _initial_values(
            expected_initial_values
        ):
            raise PascalGdnContractError("Pascal GDN metadata proof initial-state values mismatch")

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
        if _metadata_binding(slot_indices) != self._slot_binding:
            raise PascalGdnContractError("Pascal GDN slot metadata proof is stale or unbound")
        if _metadata_binding(cu_seqlens) != self._offset_binding:
            raise PascalGdnContractError("Pascal GDN offset metadata proof is stale or unbound")
        if (initial_state is None) != (self._initial_binding is None):
            raise PascalGdnContractError(
                "Pascal GDN initial-state proof requires its bound tensor"
            )
        if initial_state is not None and _metadata_binding(initial_state) != self._initial_binding:
            raise PascalGdnContractError("Pascal GDN initial-state proof is stale or unbound")
        if len(self.slot_values) != int(slot_indices.numel()):
            raise PascalGdnContractError("Pascal GDN slot metadata proof length mismatch")
        if len(self.offset_values) != int(cu_seqlens.numel()):
            raise PascalGdnContractError("Pascal GDN offset metadata proof length mismatch")
        if self.initial_values is not None and len(self.initial_values) != int(slot_indices.numel()):
            raise PascalGdnContractError("Pascal GDN initial-state proof length mismatch")
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


def _issue_pascal_gdn_metadata_proof(
    device: torch.device,
    slot_values: tuple[int, ...],
    offset_values: tuple[int, ...],
    *,
    initial_values: tuple[bool, ...] | None = None,
    owner_token: object | None = None,
    phase: str | None = None,
    pool_tensors: tuple[torch.Tensor, ...] | None = None,
) -> PascalGdnMetadataProof:
    """Issue dedicated Pascal metadata tensors from scheduler-owned host values."""

    import torch

    slots = _int32_values("Pascal GDN slot metadata", slot_values)
    offsets = _int32_values("Pascal GDN offset metadata", offset_values)
    initial = _initial_values(initial_values)
    if len(offsets) != len(slots) + 1:
        raise PascalGdnContractError("Pascal GDN offset metadata must contain B + 1 entries")
    if initial is not None and len(initial) != len(slots):
        raise PascalGdnContractError("Pascal GDN initial-state proof length mismatch")
    # Preserve version counters even when the scheduler is called from an inference-mode
    # section. These tensors are immutable launch metadata and are not the generic FLA tensors.
    with torch.inference_mode(False):
        pin = {"device": "cpu", "pin_memory": device.type == "cuda"}
        slot_host = torch.tensor(slots, dtype=torch.int32, **pin)
        offset_host = torch.tensor(offsets, dtype=torch.int32, **pin)
        initial_host = None if initial is None else torch.tensor(initial, dtype=torch.bool, **pin)
        slot_indices = slot_host.to(device, non_blocking=True)
        cu_seqlens = offset_host.to(device, non_blocking=True)
        initial_state = None if initial_host is None else initial_host.to(device, non_blocking=True)

    return PascalGdnMetadataProof(
        _METADATA_PROOF_SEAL,
        slot_indices,
        cu_seqlens,
        initial_state,
        slots,
        offsets,
        initial,
        owner_token=owner_token,
        phase=phase,
        pool_tensors=pool_tensors,
    )


def _checked_int32_tensor(name: str, value: torch.Tensor) -> list[int]:
    """Read direct-call metadata once, rejecting non-ABI integer values."""

    import torch

    if value.dtype not in (torch.int32, torch.int64):
        raise PascalGdnContractError(
            f"{name} must use int32 or int64 metadata, got {value.dtype}"
        )
    values = _cpu_int_values(name, value)
    _int32_values(name, values)
    return values


def validate_pascal_gdn_metadata(
    slot_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    num_slots: int,
    num_tokens: int,
    initial_state: torch.Tensor | None = None,
    metadata_proof: PascalGdnMetadataProof | None = None,
    phase: str | None = None,
    owner_token: object | None = None,
    pool_tensors: tuple[torch.Tensor, ...] | None = None,
    expected_slot_values: tuple[int, ...] | None = None,
    expected_offset_values: tuple[int, ...] | None = None,
    expected_initial_values: tuple[bool, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[bool] | None]:
    """Validate semantic ragged metadata before any convolution or state mutation.

    When a scheduler-issued proof is present, its dedicated tensors and immutable host values
    are returned. Generic FLA tensors remain untrusted staging metadata and are not used for
    Pascal addressing. Direct callers use synchronous host reads and retain the same checks.
    """

    import torch

    for name, value in (("slot_indices", slot_indices), ("cu_seqlens", cu_seqlens)):
        _require_tensor(name, value)
        if value.ndim != 1 or not value.is_contiguous():
            raise PascalGdnContractError(f"{name} must be a contiguous rank-1 tensor")
        if value.dtype not in (torch.int32, torch.int64):
            raise PascalGdnContractError(
                f"{name} must use int32 or int64 metadata, got {value.dtype}"
            )
    if slot_indices.device != cu_seqlens.device:
        raise PascalGdnContractError("Pascal GDN metadata tensors must share a device")
    if num_slots <= 0:
        raise PascalGdnContractError("Pascal GDN state pool must contain at least one slot")
    if num_tokens <= 0:
        raise PascalGdnContractError("Pascal GDN requires at least one token")
    if phase not in (None, "prefill", "decode"):
        raise PascalGdnContractError(f"Pascal GDN phase must be prefill or decode, got {phase!r}")

    if metadata_proof is None:
        slots = _checked_int32_tensor("slot_indices", slot_indices)
        offsets = _checked_int32_tensor("cu_seqlens", cu_seqlens)
        effective_slots = slot_indices
        effective_offsets = cu_seqlens
        if initial_state is None:
            initial = None
        else:
            if (
                initial_state.device != slot_indices.device
                or initial_state.dtype != torch.bool
                or initial_state.ndim != 1
                or not initial_state.is_contiguous()
            ):
                raise PascalGdnContractError(
                    "Pascal GDN initial-state metadata must be contiguous bool [B]"
                )
            initial = [bool(value) for value in _cpu_int_values("initial_state", initial_state)]
    else:
        if not isinstance(metadata_proof, PascalGdnMetadataProof):
            raise PascalGdnContractError("metadata_proof must be a Pascal GDN metadata proof")
        metadata_proof.validate_context(
            owner_token=owner_token,
            phase=phase,
            device=slot_indices.device,
            pool_tensors=pool_tensors,
            expected_slot_values=expected_slot_values,
            expected_offset_values=expected_offset_values,
            expected_initial_values=expected_initial_values,
        )
        effective_slots = metadata_proof.slot_indices
        effective_offsets = metadata_proof.cu_seqlens
        if (
            effective_slots.device != slot_indices.device
            or effective_offsets.device != cu_seqlens.device
            or effective_slots.dtype != torch.int32
            or effective_offsets.dtype != torch.int32
            or effective_slots.ndim != 1
            or effective_offsets.ndim != 1
            or not effective_slots.is_contiguous()
            or not effective_offsets.is_contiguous()
        ):
            raise PascalGdnContractError("Pascal GDN metadata proof has invalid effective tensors")
        slots, offsets, initial = metadata_proof.values_for(
            effective_slots, effective_offsets, metadata_proof.initial_state
        )

    if len(slots) <= 0:
        raise PascalGdnContractError("slot_indices must contain at least one request")
    if len(offsets) != len(slots) + 1:
        raise PascalGdnContractError("cu_seqlens must contain B + 1 entries")
    if len(initial or ()) not in (0, len(slots)):
        raise PascalGdnContractError("initial-state metadata must contain B entries")
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
    if phase == "decode":
        if initial is not None:
            raise PascalGdnContractError("Pascal GDN decode must not provide initial-state metadata")
        if len(slots) != num_tokens:
            raise PascalGdnContractError("Pascal GDN decode requires exactly one token per request")
        if offsets != list(range(len(slots) + 1)):
            raise PascalGdnContractError("Pascal GDN decode offsets must be exactly [0..B]")
    elif phase == "prefill" and initial is None:
        raise PascalGdnContractError("Pascal GDN prefill requires initial-state metadata")
    return effective_slots, effective_offsets, initial


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

    validate_pascal_gdn_metadata(
        slot_indices,
        cu_seqlens,
        num_slots=num_slots,
        num_tokens=num_tokens,
        metadata_proof=metadata_proof,
    )
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
        metadata_proof=metadata_proof,
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
    "pascal_gdn_recurrence",
    "validate_pascal_gdn_metadata",
    "validate_pascal_gdn_inputs",
]

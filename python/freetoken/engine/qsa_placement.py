"""Torch-free binding of QSA workspace accounting to GPU placement contracts.

The adapter derives every QSA placement byte from :func:`calculate_qsa_workspace`.
It never accepts caller-supplied QSA totals and never allocates or inspects CUDA state.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .placement_plan import (
    QSA_PERSISTENT_CATEGORIES,
    QSA_TRANSIENT_CATEGORIES,
    GPUPlacementPlan,
    PlacementInputError,
    PlacementPlan,
)
from .placement_profile import PlacementProfile

QSA_PLACEMENT_CATEGORIES = QSA_PERSISTENT_CATEGORIES + QSA_TRANSIENT_CATEGORIES
_MAX_BYTES = (1 << 63) - 1


class QSAPlacementError(PlacementInputError):
    """A QSA workspace cannot be bound to the requested placement contract."""


def _qsa_workspace_types() -> tuple[type, type, Any]:
    """Load QSA accounting through the package or its standalone torch-free seam."""
    try:
        from freetoken.attention.qsa_workspace import (
            QSAWorkspaceInputs,
            QSAWorkspacePlan,
            calculate_qsa_workspace,
        )

        return QSAWorkspaceInputs, QSAWorkspacePlan, calculate_qsa_workspace
    except ModuleNotFoundError:
        source = Path(__file__).resolve().parents[1] / "attention" / "qsa_workspace.py"
        spec = importlib.util.spec_from_file_location("freetoken._h0_qsa_workspace", source)
        if spec is None or spec.loader is None:
            raise QSAPlacementError("cannot load torch-free QSA workspace schema") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.QSAWorkspaceInputs, module.QSAWorkspacePlan, module.calculate_qsa_workspace


QSAWorkspaceInputs, QSAWorkspacePlan, calculate_qsa_workspace = _qsa_workspace_types()


def _add(values: tuple[int, ...], label: str) -> int:
    result = 0
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise QSAPlacementError(f"{label} must contain non-negative integers")
        if result > _MAX_BYTES - value:
            raise QSAPlacementError(f"{label} integer overflow")
        result += value
    return result


def _workspace_plan(value: QSAWorkspacePlan | QSAWorkspaceInputs) -> QSAWorkspacePlan:
    if isinstance(value, QSAWorkspaceInputs):
        return calculate_qsa_workspace(value)
    if not isinstance(value, QSAWorkspacePlan):
        raise QSAPlacementError(
            "QSA workspace must be QSAWorkspaceInputs or a calculated QSAWorkspacePlan"
        )
    try:
        recalculated = calculate_qsa_workspace(value.request)
    except (TypeError, ValueError) as exc:
        raise QSAPlacementError(f"QSA workspace plan request is invalid: {exc}") from exc
    if (
        value.inventory != recalculated.inventory
        or value.required_bytes != recalculated.required_bytes
        or value.persistent_bytes != recalculated.persistent_bytes
        or value.capture_resident_bytes != recalculated.capture_resident_bytes
        or value.eager_transient_peak_bytes != recalculated.eager_transient_peak_bytes
    ):
        raise QSAPlacementError(
            "QSA workspace plan contains caller-supplied or inconsistent derived bytes"
        )
    return recalculated


def derive_qsa_placement_categories(
    workspace: QSAWorkspacePlan | QSAWorkspaceInputs,
) -> Mapping[str, int]:
    """Project calculated QSA components into #73's exact ten placement categories.

    The compressed slab, pending ring and index RoPE are retained state and therefore occupy
    ``qsa_persistent_state``.  Every other calculated component is a category-specific transient
    envelope, while the workspace plan retains the exact phase live/peak overlap separately.
    """
    plan = _workspace_plan(workspace)
    state = plan.inventory["state"]
    transient_state = state.bytes - plan.persistent_bytes
    if transient_state < 0:
        raise QSAPlacementError("QSA persistent state exceeds its calculated state category")
    values = {
        "qsa_persistent_score": 0,
        "qsa_persistent_top_k": 0,
        "qsa_persistent_expand_gather": 0,
        "qsa_persistent_attention": 0,
        "qsa_persistent_state": plan.persistent_bytes,
        "qsa_transient_score": plan.inventory["score"].bytes,
        "qsa_transient_top_k": plan.inventory["top_k"].bytes,
        "qsa_transient_expand_gather": plan.inventory["expand_gather"].bytes,
        "qsa_transient_attention": plan.inventory["attention"].bytes,
        "qsa_transient_state": transient_state,
    }
    if tuple(values) != QSA_PLACEMENT_CATEGORIES:
        raise QSAPlacementError("QSA placement category projection is incomplete")
    _add(tuple(values.values()), "QSA placement categories")
    return MappingProxyType(values)


def _request_geometry(plan: QSAWorkspacePlan) -> dict[str, Any]:
    return {name: getattr(plan.request, name) for name in plan.request.__dataclass_fields__}


def _validate_profile_geometry(plan: QSAWorkspacePlan, profile: PlacementProfile) -> None:
    if not isinstance(profile, PlacementProfile):
        raise QSAPlacementError("profile must be a PlacementProfile")
    expected = _request_geometry(plan)
    actual = dict(profile.identity.qsa_geometry)
    if actual != expected:
        differing = sorted(
            name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name)
        )
        raise QSAPlacementError(f"QSA workspace/profile geometry mismatch: {differing}")


def _validate_gpu_categories(
    categories: Mapping[str, int], expected: Mapping[str, int], rank: int
) -> None:
    for name in QSA_PLACEMENT_CATEGORIES:
        if categories[name] != expected[name]:
            raise QSAPlacementError(
                f"GPU {rank} {name} does not match calculated QSA workspace bytes: "
                f"planned={categories[name]}, calculated={expected[name]}"
            )


@dataclass(frozen=True, slots=True)
class QSAPlacementBinding:
    """Immutable QSA workspace binding and its validated placement targets."""

    workspace: QSAWorkspacePlan
    categories: Mapping[str, int]
    gpu_plans: tuple[GPUPlacementPlan, ...]
    profile_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, QSAWorkspacePlan):
            raise QSAPlacementError("binding workspace must be a QSAWorkspacePlan")
        workspace = _workspace_plan(self.workspace)
        expected = derive_qsa_placement_categories(self.workspace)
        if not isinstance(self.categories, Mapping) or dict(self.categories) != dict(expected):
            raise QSAPlacementError("binding categories must be the calculated QSA projection")
        try:
            targets = tuple(self.gpu_plans)
        except TypeError as exc:
            raise QSAPlacementError("binding GPU plans must be iterable") from exc
        if not targets or any(not isinstance(item, GPUPlacementPlan) for item in targets):
            raise QSAPlacementError("binding GPU plans must contain GPUPlacementPlan values")
        for gpu in targets:
            _validate_gpu_categories(gpu.categories, expected, gpu.rank)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "categories", MappingProxyType(dict(expected)))
        object.__setattr__(self, "gpu_plans", targets)
        if self.profile_digest is not None and (
            not isinstance(self.profile_digest, str) or len(self.profile_digest) != 64
        ):
            raise QSAPlacementError("binding profile_digest must be a SHA-256 digest")

    @property
    def persistent_bytes(self) -> int:
        return self.workspace.persistent_bytes

    @property
    def live_bytes(self) -> int:
        """Return the exact calculated live high-water for the active workspace phase."""
        transient = (
            self.workspace.capture_resident_bytes
            if self.workspace.request.phase == "capture"
            else self.workspace.eager_transient_peak_bytes
        )
        return _add((self.persistent_bytes, transient), "QSA live bytes")

    @property
    def peak_bytes(self) -> int:
        """Return the exact calculated persistent-plus-phase-peak requirement."""
        return self.workspace.required_bytes

    @property
    def placement_categories(self) -> Mapping[str, int]:
        return self.categories

    @property
    def qsa_categories(self) -> Mapping[str, int]:
        return self.categories

    @property
    def placement_qsa_required_bytes(self) -> int:
        """Return the conservative sum of the ten separately observable placement buckets."""
        return _add(tuple(self.categories.values()), "QSA placement categories")

    @property
    def gpu_plan(self) -> GPUPlacementPlan:
        if len(self.gpu_plans) != 1:
            raise QSAPlacementError("binding contains more than one GPU plan")
        return self.gpu_plans[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "categories": dict(self.categories),
            "persistent_bytes": self.persistent_bytes,
            "live_bytes": self.live_bytes,
            "peak_bytes": self.peak_bytes,
            "placement_qsa_required_bytes": self.placement_qsa_required_bytes,
            "gpu_keys": [item.key for item in self.gpu_plans],
            "profile_digest": self.profile_digest,
            "workspace": self.workspace.as_dict(),
        }


def bind_qsa_workspace(
    workspace: QSAWorkspacePlan | QSAWorkspaceInputs,
    target: GPUPlacementPlan | PlacementPlan | PlacementProfile,
    *,
    profile: PlacementProfile | None = None,
) -> QSAPlacementBinding:
    """Validate a calculated QSA workspace against a GPU plan or placement profile.

    A profile target validates its canonical QSA geometry and embedded plan.  A direct plan
    target may be paired with ``profile`` for the same geometry check.  All QSA categories must
    exactly equal the projection derived from the workspace request; extra caller bytes are not
    accepted as a substitute for calculation.
    """
    plan = _workspace_plan(workspace)
    profile_digest: str | None = None
    if isinstance(target, PlacementProfile):
        if profile is not None:
            raise QSAPlacementError("profile must not be supplied twice")
        profile = target
        target_plan: PlacementPlan = target.plan
    elif isinstance(target, GPUPlacementPlan):
        target_plan = None
    elif isinstance(target, PlacementPlan):
        target_plan = target
    else:
        raise QSAPlacementError(
            "target must be a GPUPlacementPlan, PlacementPlan, or PlacementProfile"
        )
    if profile is not None:
        _validate_profile_geometry(plan, profile)
        profile_digest = profile.digest
        if isinstance(target, GPUPlacementPlan):
            targets = (target,)
            if profile.plan.gpu_count != 1 or profile.plan.gpus[0] != target:
                raise QSAPlacementError("GPU plan does not match the supplied placement profile")
        else:
            targets = profile.plan.gpus if target_plan is None else target_plan.gpus
            if target_plan is not None and target_plan != profile.plan:
                raise QSAPlacementError("placement plan does not match the supplied profile")
    elif isinstance(target, GPUPlacementPlan):
        targets = (target,)
    else:
        targets = target_plan.gpus
    expected = derive_qsa_placement_categories(plan)
    for gpu in targets:
        _validate_gpu_categories(gpu.categories, expected, gpu.rank)
    return QSAPlacementBinding(plan, expected, targets, profile_digest)


validate_qsa_workspace_placement = bind_qsa_workspace
bind_qsa_workspace_to_placement = bind_qsa_workspace
qsa_placement_categories = derive_qsa_placement_categories
adapt_qsa_workspace_to_placement = bind_qsa_workspace


__all__ = [
    "QSA_PLACEMENT_CATEGORIES",
    "QSAPlacementBinding",
    "QSAPlacementError",
    "adapt_qsa_workspace_to_placement",
    "bind_qsa_workspace",
    "bind_qsa_workspace_to_placement",
    "derive_qsa_placement_categories",
    "qsa_placement_categories",
    "validate_qsa_workspace_placement",
]

"""Torch-free binding of QSA workspace accounting to GPU placement contracts.

The adapter derives every QSA placement byte from :func:`calculate_qsa_workspace`.
It never accepts caller-supplied QSA totals and never allocates or inspects CUDA state.
"""

from __future__ import annotations

import importlib.util
import re
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QSAPlacementError(PlacementInputError):
    """A QSA workspace cannot be bound to the requested placement contract."""


def _qsa_workspace_types() -> tuple[type, type, Any, Any]:
    """Load QSA accounting without importing the attention package or Torch."""
    module = sys.modules.get("freetoken.attention.qsa_workspace")
    if module is None:
        module = sys.modules.get("freetoken._h0_qsa_workspace")
    if module is None:
        source = Path(__file__).resolve().parents[1] / "attention" / "qsa_workspace.py"
        spec = importlib.util.spec_from_file_location("freetoken._h0_qsa_workspace", source)
        if spec is None or spec.loader is None:
            raise QSAPlacementError("cannot load torch-free QSA workspace schema") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    try:
        return (
            module.QSAWorkspaceInputs,
            module.QSAWorkspacePlan,
            module.calculate_qsa_workspace,
            module.qsa_topk_scratch_width,
        )
    except AttributeError as exc:
        raise QSAPlacementError("QSA workspace module has an incomplete schema") from exc


(
    QSAWorkspaceInputs,
    QSAWorkspacePlan,
    calculate_qsa_workspace,
    qsa_topk_scratch_width,
) = _qsa_workspace_types()


def _add(values: tuple[int, ...], label: str) -> int:
    result = 0
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise QSAPlacementError(f"{label} must contain non-negative integers")
        if result > _MAX_BYTES - value:
            raise QSAPlacementError(f"{label} integer overflow")
        result += value
    return result


def _mul(values: tuple[int, ...], label: str) -> int:
    result = 1
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise QSAPlacementError(f"{label} must contain non-negative integers")
        if value and result > _MAX_BYTES // value:
            raise QSAPlacementError(f"{label} integer overflow")
        result *= value
    return result


def _workspace_plan(value: QSAWorkspacePlan | QSAWorkspaceInputs) -> QSAWorkspacePlan:
    if isinstance(value, QSAWorkspaceInputs):
        return calculate_qsa_workspace(value)
    if _is_runtime_inputs(value):
        return calculate_qsa_workspace(_coerce_workspace_inputs(value))
    if isinstance(value, QSAWorkspacePlan):
        request = value.request
    elif _is_runtime_plan(value):
        request = _coerce_workspace_inputs(value.request)
    else:
        raise QSAPlacementError(
            "QSA workspace must be QSAWorkspaceInputs or a calculated QSAWorkspacePlan"
        )
    try:
        recalculated = calculate_qsa_workspace(request)
    except (TypeError, ValueError) as exc:
        raise QSAPlacementError(f"QSA workspace plan request is invalid: {exc}") from exc
    if (
        _inventory_payload(value.inventory) != _inventory_payload(recalculated.inventory)
        or value.required_bytes != recalculated.required_bytes
        or value.persistent_bytes != recalculated.persistent_bytes
        or value.capture_resident_bytes != recalculated.capture_resident_bytes
        or value.eager_transient_peak_bytes != recalculated.eager_transient_peak_bytes
        or _telemetry_payload(value.telemetry) != _telemetry_payload(recalculated.telemetry)
    ):
        raise QSAPlacementError(
            "QSA workspace plan contains caller-supplied or inconsistent derived bytes"
        )
    return recalculated


def _canonical_qsa_module() -> Any | None:
    return sys.modules.get("freetoken.attention.qsa_workspace")


def _is_runtime_plan(value: Any) -> bool:
    module = _canonical_qsa_module()
    return module is not None and isinstance(value, getattr(module, "QSAWorkspacePlan", ()))


def _is_runtime_inputs(value: Any) -> bool:
    module = _canonical_qsa_module()
    return module is not None and isinstance(value, getattr(module, "QSAWorkspaceInputs", ()))


def _coerce_workspace_inputs(value: Any) -> QSAWorkspaceInputs:
    if isinstance(value, QSAWorkspaceInputs):
        return value
    module = _canonical_qsa_module()
    runtime_type = getattr(module, "QSAWorkspaceInputs", ()) if module is not None else ()
    if runtime_type and isinstance(value, runtime_type):
        try:
            return QSAWorkspaceInputs(
                **{name: getattr(value, name) for name in QSAWorkspaceInputs.__dataclass_fields__}
            )
        except (TypeError, ValueError) as exc:
            raise QSAPlacementError(f"QSA workspace request is invalid: {exc}") from exc
    raise QSAPlacementError("QSA workspace request must be a calculated QSAWorkspaceInputs")


def _inventory_payload(inventory: Any) -> dict[str, Any]:
    try:
        names = tuple(inventory)
        return {
            name: {
                "name": inventory[name].name,
                "bytes": inventory[name].bytes,
                "components": dict(inventory[name].components),
                "shapes": {
                    component: tuple(shape) for component, shape in inventory[name].shapes.items()
                },
            }
            for name in names
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise QSAPlacementError("QSA workspace inventory is malformed") from exc


def _telemetry_payload(telemetry: Any) -> Any:
    try:
        return telemetry.as_dict()
    except AttributeError as exc:
        raise QSAPlacementError("QSA workspace telemetry is malformed") from exc


def _category_totals(
    *,
    score: int = 0,
    top_k: int = 0,
    expand_gather: int = 0,
    attention: int = 0,
    state: int = 0,
) -> dict[str, int]:
    return {
        "score": score,
        "top_k": top_k,
        "expand_gather": expand_gather,
        "attention": attention,
        "state": state,
    }


def _eager_transient_categories(plan: QSAWorkspacePlan) -> dict[str, int]:
    """Return category attribution for the phase with the largest eager live set."""
    score = plan.inventory["score"].components
    expand = plan.inventory["expand_gather"].components
    attention = plan.inventory["attention"].bytes
    state = plan.inventory["state"].components
    metadata = _add(
        tuple(
            score[name]
            for name in (
                "last_indices",
                "token_to_req",
                "cu_seqlens",
                "seq_lens",
                "ring_slots",
                "block_table",
            )
        ),
        "QSA eager metadata",
    )
    scatter = _add((state["cmp_rows"], state["ring_rows"]), "QSA eager scatter rows")
    index_categories = _category_totals(
        score=metadata,
        state=_add(
            (state["pooled"], state["first_positions"], scatter),
            "QSA eager index-update state",
        ),
    )
    selection_categories = _category_totals(
        score=_add(
            (
                metadata,
                score["q_index"],
                score["logits"],
                score["visible"],
                score.get("q_index_fp32", 0),
                score.get("request_keys", 0),
                score.get("request_keys_fp32", 0),
                score.get("vector_score_heads", 0),
            ),
            "QSA eager selection score",
        ),
        top_k=plan.inventory["top_k"].bytes,
        expand_gather=expand["indices"],
        state=scatter,
    )
    attention_categories = _category_totals(
        score=metadata,
        expand_gather=expand["indices"],
        attention=attention,
        state=scatter,
    )
    phases = (
        (
            "index-update",
            _add(tuple(index_categories.values()), "QSA eager index-update phase"),
            index_categories,
        ),
        (
            "selection",
            _add(tuple(selection_categories.values()), "QSA eager selection phase"),
            selection_categories,
        ),
        (
            "attention",
            _add(tuple(attention_categories.values()), "QSA eager attention phase"),
            attention_categories,
        ),
    )
    phase = max(phases, key=lambda item: item[1])
    expected = plan.eager_transient_peak_bytes
    if (
        phase[1] != expected
        or _add(tuple(phase[2].values()), "QSA eager category total") != expected
    ):
        raise QSAPlacementError(
            f"QSA eager phase attribution disagrees with calculated peak: "
            f"phase={phase[0]}, calculated={phase[1]}, expected={expected}"
        )
    return phase[2]


def _capture_categories(plan: QSAWorkspacePlan) -> tuple[dict[str, int], dict[str, int]]:
    """Return persistent and transient category attribution for capture allocations."""
    request = plan.request
    attention = plan.inventory["attention"].bytes
    state = plan.inventory["state"].components
    capture_bs = request.capture_max_batch_size
    page_count = request.page_count
    block_top_k = request.top_k // request.compression_ratio
    graph_metadata = _add(
        (
            _mul((capture_bs, page_count, 4), "QSA capture block table"),
            _mul((capture_bs, 4), "QSA capture kv lengths"),
            _mul((capture_bs, 4), "QSA capture table index"),
            _mul((capture_bs, 4), "QSA capture token-to-request"),
            _mul((capture_bs + 1, 4), "QSA capture cu seqlens"),
        ),
        "QSA capture graph metadata",
    )
    # Graph buffers are retained for replay, while the active attention output and metadata remain
    # transient around each replay.  Their category partition must sum to the calculator's
    # capture high-water rather than adding mutually exclusive eager phases.
    persistent = _category_totals(
        score=_add(
            (
                graph_metadata,
                _mul(
                    (request.capture_chunk_rows, request.score_columns, 4),
                    "QSA capture logits",
                ),
                _mul((capture_bs, 4), "QSA capture visible blocks"),
                _mul(
                    (capture_bs, request.index_heads, request.index_head_dim, 2),
                    "QSA capture index query",
                ),
            ),
            "QSA capture score buffers",
        ),
        top_k=_add(
            (
                _mul((capture_bs, block_top_k, 4), "QSA capture top-k blocks"),
                _mul(
                    (
                        capture_bs,
                        qsa_topk_scratch_width(
                            request.score_columns, block_top_k, request.topk_backend
                        ),
                        4,
                    ),
                    "QSA capture top-k scratch",
                ),
            ),
            "QSA capture top-k buffers",
        ),
        expand_gather=_mul(
            (capture_bs, request.selection_width, 4), "QSA capture expanded indices"
        ),
        state=_add(
            (
                plan.persistent_bytes,
                _mul((capture_bs, request.index_head_dim, 2), "QSA capture pooled rows"),
                _mul((capture_bs, 4), "QSA capture first positions"),
            ),
            "QSA capture state buffers",
        ),
    )
    transient = _category_totals(
        score=_mul((request.batch_size, 4), "QSA capture last indices"),
        attention=attention,
        state=_add((state["cmp_rows"], state["ring_rows"]), "QSA capture scatter rows"),
    )
    if (
        _add(tuple(persistent.values()) + tuple(transient.values()), "QSA capture category total")
        != plan.required_bytes
    ):
        raise QSAPlacementError(
            "QSA capture category attribution disagrees with calculated high-water"
        )
    return persistent, transient


def derive_qsa_placement_categories(
    workspace: QSAWorkspacePlan | QSAWorkspaceInputs,
) -> Mapping[str, int]:
    """Project calculated QSA components into #73's exact ten placement categories.

    Eager plans attribute the winning live phase to the score, top-k, expand-gather, attention,
    and state buckets without adding mutually exclusive phases. Capture plans attribute retained
    graph buffers to persistent buckets and active replay buffers to transient buckets.
    """
    plan = _workspace_plan(workspace)
    if plan.request.phase == "capture":
        persistent, transient = _capture_categories(plan)
    else:
        persistent = _category_totals(state=plan.persistent_bytes)
        transient = _eager_transient_categories(plan)
    values = {
        **{
            f"qsa_persistent_{name}": persistent[name]
            for name in ("score", "top_k", "expand_gather", "attention", "state")
        },
        **{
            f"qsa_transient_{name}": transient[name]
            for name in ("score", "top_k", "expand_gather", "attention", "state")
        },
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
        if not isinstance(self.workspace, QSAWorkspacePlan) and not _is_runtime_plan(
            self.workspace
        ):
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
        if len(targets) not in {1, 2}:
            raise QSAPlacementError("binding requires exactly one or two GPU plans")
        if tuple(item.rank for item in targets) != tuple(range(len(targets))):
            raise QSAPlacementError("binding GPU ranks must be contiguous and ordered")
        if len({item.gpu_uuid for item in targets}) != len(targets):
            raise QSAPlacementError("binding GPU UUIDs must be unique")
        if len(targets) == 2:
            raise QSAPlacementError(
                "dual-GPU QSA binding requires an explicit ownership/partition policy"
            )
        for gpu in targets:
            if gpu.status != "ready":
                raise QSAPlacementError(
                    f"GPU {gpu.rank} placement plan is not capacity-safe: {gpu.status}"
                )
            _validate_gpu_categories(gpu.categories, expected, gpu.rank)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "categories", MappingProxyType(dict(expected)))
        object.__setattr__(self, "gpu_plans", targets)
        if self.profile_digest is not None and (
            not isinstance(self.profile_digest, str)
            or not _SHA256_RE.fullmatch(self.profile_digest)
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
        """Return the exact sum of the ten separately observable placement buckets."""
        return _add(tuple(self.categories.values()), "QSA placement categories")

    @property
    def placement_persistent_bytes(self) -> int:
        return _add(
            tuple(self.categories[name] for name in QSA_PERSISTENT_CATEGORIES),
            "QSA placement persistent categories",
        )

    @property
    def placement_transient_high_water_bytes(self) -> int:
        return _add(
            tuple(self.categories[name] for name in QSA_TRANSIENT_CATEGORIES),
            "QSA placement transient categories",
        )

    @property
    def gpu_plan(self) -> GPUPlacementPlan:
        if len(self.gpu_plans) != 1:
            raise QSAPlacementError("binding contains more than one GPU plan")
        return self.gpu_plans[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "categories": dict(self.categories),
            "persistent_bytes": self.persistent_bytes,
            "placement_persistent_bytes": self.placement_persistent_bytes,
            "placement_transient_high_water_bytes": self.placement_transient_high_water_bytes,
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
        if gpu.status != "ready":
            raise QSAPlacementError(
                f"GPU {gpu.rank} placement plan is not capacity-safe: {gpu.status}"
            )
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

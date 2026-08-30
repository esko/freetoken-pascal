"""Engine public API with a Torch-free import seam for accounting utilities.

The placement planner is intentionally usable in hosted H0 tooling where Torch is not
installed.  Keep the runtime objects lazy so importing ``freetoken.engine.placement_plan``
does not eagerly import the CUDA-serving engine.
"""

from typing import TYPE_CHECKING

from .placement_plan import (
    PLACEMENT_CATEGORIES,
    BackoffDecision,
    BackoffProfile,
    BackoffStateMachine,
    CanaryResult,
    PlacementObservation,
    PlacementPlan,
    PlacementPlanInput,
    PlacementPlannerError,
    evaluate_canary,
    plan_placement,
)
from .placement_profile import (
    PLACEMENT_PROFILE_IDENTITY_SCHEMA_NAME,
    PLACEMENT_PROFILE_SCHEMA_NAME,
    PLACEMENT_PROFILE_SCHEMA_VERSION,
    GPUProfileTopology,
    PlacementProfile,
    PlacementProfileIdentity,
    canonical_json_bytes,
)
from .qsa_placement import (
    QSA_PLACEMENT_CATEGORIES,
    QSAPlacementBinding,
    QSAPlacementError,
    adapt_qsa_workspace_to_placement,
    bind_qsa_workspace,
    derive_qsa_placement_categories,
    validate_qsa_workspace_placement,
)

if TYPE_CHECKING:
    from .config import EngineConfig
    from .engine import Engine, ForwardOutput
    from .sample import BatchSamplingArgs

__all__ = [
    "PLACEMENT_CATEGORIES",
    "PLACEMENT_PROFILE_IDENTITY_SCHEMA_NAME",
    "PLACEMENT_PROFILE_SCHEMA_NAME",
    "PLACEMENT_PROFILE_SCHEMA_VERSION",
    "QSA_PLACEMENT_CATEGORIES",
    "BackoffDecision",
    "BackoffProfile",
    "BackoffStateMachine",
    "BatchSamplingArgs",
    "CanaryResult",
    "Engine",
    "EngineConfig",
    "ForwardOutput",
    "GPUProfileTopology",
    "PlacementObservation",
    "PlacementPlan",
    "PlacementPlanInput",
    "PlacementPlannerError",
    "PlacementProfile",
    "PlacementProfileIdentity",
    "QSAPlacementBinding",
    "QSAPlacementError",
    "adapt_qsa_workspace_to_placement",
    "bind_qsa_workspace",
    "canonical_json_bytes",
    "derive_qsa_placement_categories",
    "evaluate_canary",
    "plan_placement",
    "validate_qsa_workspace_placement",
]


def __getattr__(name: str):
    if name == "EngineConfig":
        from .config import EngineConfig

        return EngineConfig
    if name in {"Engine", "ForwardOutput"}:
        from .engine import Engine, ForwardOutput

        return {"Engine": Engine, "ForwardOutput": ForwardOutput}[name]
    if name == "BatchSamplingArgs":
        from .sample import BatchSamplingArgs

        return BatchSamplingArgs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

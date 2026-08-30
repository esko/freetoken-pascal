"""H0 contract tests for Qwen4 GatedDeltaNet backend selection."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_PATH = ROOT / "python/freetoken/models/qwen4_exp/gdn_contract.py"
_SPEC = importlib.util.spec_from_file_location("qwen4_gdn_contract_h0", _CONTRACT_PATH)
assert _SPEC and _SPEC.loader
_CONTRACT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONTRACT
_SPEC.loader.exec_module(_CONTRACT)
GdnDispatchDecision = _CONTRACT.GdnDispatchDecision
GdnDispatchError = _CONTRACT.GdnDispatchError
parse_gdn_mode = _CONTRACT.parse_gdn_mode
resolve_gdn_dispatch = _CONTRACT.resolve_gdn_dispatch


def test_gdn_contract_is_torch_free() -> None:
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported_names = {alias.name.split(".", 1)[0] for node in imports for alias in node.names}
    assert "torch" not in imported_names


def test_gdn_decision_is_immutable_and_serializable() -> None:
    decision = resolve_gdn_dispatch(
        requested_mode="auto",
        capability=(6, 1),
        dtype="bfloat16",
        fla_available=True,
        triton_candidate_available=True,
    )

    assert isinstance(decision, GdnDispatchDecision)
    assert dataclasses.is_dataclass(decision)
    assert decision.requested_mode == "auto"
    assert decision.selected_implementation == "torch-reference"
    assert decision.capability == (6, 1)
    assert decision.dtype == "bfloat16"
    assert decision.fla_available is True
    assert decision.triton_candidate_available is True
    assert decision.fallback_reason == "unsupported-capability"
    assert decision.as_dict()["selected_implementation"] == "torch-reference"
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.capability = (7, 0)


@pytest.mark.parametrize(
    ("capability", "dtype", "fla_available", "reason"),
    [
        ((6, 1), "bfloat16", True, "unsupported-capability"),
        ((8, 0), "float64", True, "unsupported-dtype"),
        ((8, 0), "bfloat16", False, "fla-unavailable"),
    ],
)
def test_auto_falls_back_to_reference_with_visible_reason(
    capability, dtype, fla_available, reason
) -> None:
    decision = resolve_gdn_dispatch(
        requested_mode="auto",
        capability=capability,
        dtype=dtype,
        fla_available=fla_available,
        triton_candidate_available=True,
    )

    assert decision.selected_implementation == "torch-reference"
    assert decision.fallback_reason == reason


def test_auto_preserves_qualified_modern_fla_path() -> None:
    decision = resolve_gdn_dispatch(
        requested_mode="auto",
        capability=(8, 0),
        dtype="bfloat16",
        fla_available=True,
        triton_candidate_available=True,
    )

    assert decision.selected_implementation == "fla"
    assert decision.fallback_reason is None


def test_reference_mode_is_always_available() -> None:
    decision = resolve_gdn_dispatch(
        requested_mode="reference",
        capability=(8, 0),
        dtype="float64",
        fla_available=False,
        triton_candidate_available=False,
    )

    assert decision.requested_mode == "torch-reference"
    assert decision.selected_implementation == "torch-reference"
    assert decision.fallback_reason is None


def test_forced_candidate_fails_closed_when_unavailable() -> None:
    with pytest.raises(GdnDispatchError, match="candidate-unavailable"):
        resolve_gdn_dispatch(
            requested_mode="triton-candidate",
            capability=(8, 0),
            dtype="bfloat16",
            fla_available=True,
            triton_candidate_available=False,
        )


def test_auto_never_selects_pascal_even_when_the_explicit_gate_is_positive() -> None:
    decision = resolve_gdn_dispatch(
        requested_mode="auto",
        capability=(6, 1),
        dtype="float32",
        fla_available=True,
        triton_candidate_available=True,
        pascal_fp32_available=True,
    )

    assert decision.selected_implementation == "torch-reference"
    assert decision.fallback_reason == "unsupported-capability"
    assert decision.pascal_fp32_available is True


def test_pascal_backend_is_explicit_and_requires_positive_qualification_gate() -> None:
    with pytest.raises(GdnDispatchError, match="pascal-fp32-unqualified"):
        resolve_gdn_dispatch(
            requested_mode="pascal-fp32",
            capability=(6, 1),
            dtype="float32",
            pascal_fp32_available=False,
        )

    decision = resolve_gdn_dispatch(
        requested_mode="pascal-fp32",
        capability=(6, 1),
        dtype="float32",
        pascal_fp32_available=True,
    )
    assert decision.selected_implementation == "pascal-fp32"
    assert decision.fallback_reason is None


@pytest.mark.parametrize(
    ("capability", "dtype", "message"),
    [((7, 0), "float32", "requires sm_61"), ((6, 1), "bfloat16", "requires float32")],
)
def test_pascal_backend_rejects_unqualified_inputs(capability, dtype, message) -> None:
    with pytest.raises(GdnDispatchError, match=message):
        resolve_gdn_dispatch(
            requested_mode="pascal-fp32",
            capability=capability,
            dtype=dtype,
            pascal_fp32_available=True,
        )


@pytest.mark.parametrize("mode", ["", "bogus", " Auto "])
def test_invalid_gdn_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported GDN mode"):
        parse_gdn_mode(mode)


def test_gdn_forward_contract_is_observable_before_fla_import() -> None:
    source = (ROOT / "python/freetoken/models/qwen4_exp/gdn.py").read_text(encoding="utf-8")
    assert (
        "from freetoken.models.qwen3_5_moe.gdn_kernels import"
        not in source.split("class Qwen4ExpGatedDeltaNet", 1)[0]
    )
    assert source.index("resolve_gdn_dispatch(") < source.index("gdn_decode_fla(")
    assert source.index("resolve_gdn_dispatch(") < source.index("gdn_prefill_chunk_fla(")
    assert "pascal_fp32_available=self._gdn_pascal_available" in source


def test_gdn_observer_receives_the_immutable_decision() -> None:
    observed = []
    decision = resolve_gdn_dispatch(
        requested_mode="auto",
        capability=(6, 1),
        dtype="bfloat16",
        fla_available=True,
        triton_candidate_available=True,
        observer=observed.append,
    )

    assert observed == [decision]

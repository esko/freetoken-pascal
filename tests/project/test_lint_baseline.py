from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_lint_baseline.py"
SPEC = importlib.util.spec_from_file_location("check_lint_baseline", SCRIPT)
assert SPEC and SPEC.loader
CHECK_LINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_LINT)


def test_imported_source_lint_baseline_has_not_regressed() -> None:
    baseline = json.loads((ROOT / "manifests" / "ruff-baseline.json").read_text(encoding="utf-8"))

    assert CHECK_LINT.validate_baseline(baseline) == []

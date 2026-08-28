from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "manifests" / "ruff-baseline.json"
SOURCE_ROOTS = ("python", "tests", "benchmarks")


def _ruff_version() -> str:
    result = subprocess.run(["ruff", "--version"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def collect_violations() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ruff", "check", *SOURCE_ROOTS, "--output-format", "json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"ruff failed with exit {result.returncode}: {result.stderr.strip()}")
    diagnostics = json.loads(result.stdout)
    violations = []
    for diagnostic in diagnostics:
        filename = Path(diagnostic["filename"])
        try:
            filename = filename.resolve().relative_to(ROOT)
        except ValueError as error:
            raise RuntimeError(
                f"ruff reported a path outside the repository: {filename}"
            ) from error
        violations.append(
            {
                "path": filename.as_posix(),
                "code": diagnostic["code"],
                "row": diagnostic["location"]["row"],
                "column": diagnostic["location"]["column"],
                "message": diagnostic["message"],
            }
        )
    return sorted(
        violations,
        key=lambda item: (item["path"], item["row"], item["column"], item["code"]),
    )


def build_baseline() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ruff_version": _ruff_version(),
        "source_roots": list(SOURCE_ROOTS),
        "violations": collect_violations(),
    }


def validate_baseline(baseline: Any) -> list[str]:
    if not isinstance(baseline, dict) or baseline.get("schema_version") != 1:
        return ["lint baseline must be an object with schema_version 1"]
    if baseline.get("ruff_version") != _ruff_version():
        return [
            f"lint baseline expects {baseline.get('ruff_version')!r}, found {_ruff_version()!r}"
        ]
    if baseline.get("source_roots") != list(SOURCE_ROOTS):
        return [f"lint baseline source_roots must be {list(SOURCE_ROOTS)!r}"]
    expected = baseline.get("violations")
    if not isinstance(expected, list):
        return ["lint baseline violations must be an array"]
    actual = collect_violations()
    if expected == actual:
        return []
    expected_rows = {json.dumps(item, sort_keys=True) for item in expected}
    actual_rows = {json.dumps(item, sort_keys=True) for item in actual}
    added = sorted(actual_rows - expected_rows)
    removed = sorted(expected_rows - actual_rows)
    errors = []
    if added:
        errors.append(f"{len(added)} new or moved lint violation(s); first: {added[0]}")
    if removed:
        errors.append(f"{len(removed)} resolved or moved lint violation(s); regenerate baseline")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce the imported source lint baseline")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        args.baseline.write_text(
            json.dumps(build_baseline(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote lint baseline to {args.baseline}")
        return 0
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: unable to read lint baseline: {error}", file=sys.stderr)
        return 1
    errors = validate_baseline(baseline)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"validated source-wide lint baseline ({len(baseline['violations'])} violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

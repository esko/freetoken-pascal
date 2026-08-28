from __future__ import annotations

import argparse
import hashlib
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
    violations = collect_violations()
    fingerprint = hashlib.sha256(
        json.dumps(violations, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "ruff_version": _ruff_version(),
        "source_roots": list(SOURCE_ROOTS),
        "violation_count": len(violations),
        "violations_sha256": fingerprint,
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
    actual = build_baseline()
    if baseline == actual:
        return []
    return [
        "source-wide lint findings changed: "
        f"expected {baseline.get('violation_count')} findings with fingerprint "
        f"{baseline.get('violations_sha256')!r}, found {actual['violation_count']} with "
        f"fingerprint {actual['violations_sha256']!r}; review Ruff output and regenerate baseline"
    ]


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
    print(f"validated source-wide lint baseline ({baseline['violation_count']} violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

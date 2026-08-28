from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "upstreams.yaml"


def resolve_remote_ref(repository: str, upstream_ref: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repository, upstream_ref],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError(f"expected one remote ref for {repository} {upstream_ref}")
    return rows[0][0]


def classify_pin(pinned: str, current: str | None) -> str:
    if current is None:
        return "offline"
    if current == pinned:
        return "current"
    return "changed"


def build_report(data: dict[str, Any], *, offline: bool = False) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for source in data["sources"]:
        current: str | None = None
        error: str | None = None
        if not offline:
            try:
                current = resolve_remote_ref(source["repository"], source["upstream_ref"])
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                error = str(exc)
        status = "unavailable" if error else classify_pin(source["ref"], current)
        compare_url = None
        if current and current != source["ref"]:
            compare_url = f"{source['repository']}/compare/{source['ref']}...{current}"
        report.append(
            {
                "id": source["id"],
                "usage": source["usage"],
                "pinned": source["ref"],
                "upstream_ref": source["upstream_ref"],
                "current": current,
                "status": status,
                "compare_url": compare_url,
                "error": error,
            }
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report changes from pinned upstream revisions")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        report = build_report(data, offline=args.offline)
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: unable to build upstream report: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"sources": report}, indent=2, sort_keys=True))
    else:
        for row in report:
            current = row["current"] or "-"
            print(f"{row['id']}: {row['status']} pinned={row['pinned']} current={current}")
            if row["compare_url"]:
                print(f"  compare: {row['compare_url']}")
            if row["error"]:
                print(f"  error: {row['error']}")
    return 1 if any(row["status"] == "unavailable" for row in report) else 0


if __name__ == "__main__":
    raise SystemExit(main())

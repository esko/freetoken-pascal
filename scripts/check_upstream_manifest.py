from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "upstreams.yaml"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_PLACEHOLDER = "TO_BE_PINNED"


def main() -> int:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen: set[str] = set()

    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    for source in data.get("sources", []):
        source_id = source.get("id")
        if not source_id or source_id in seen:
            errors.append(f"invalid or duplicate source id: {source_id!r}")
        seen.add(source_id)
        for key in ("repository", "ref", "license", "role", "imports"):
            if key not in source:
                errors.append(f"{source_id}: missing {key}")
        ref = source.get("ref", "")
        if ref != ALLOWED_PLACEHOLDER and not SHA_RE.fullmatch(ref):
            errors.append(f"{source_id}: ref must be a 40-char SHA or {ALLOWED_PLACEHOLDER}")
        if not str(source.get("repository", "")).startswith("https://github.com/"):
            errors.append(f"{source_id}: repository must be a GitHub URL")
        if not isinstance(source.get("imports"), list):
            errors.append(f"{source_id}: imports must be a list")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1

    placeholders = [
        source["id"] for source in data["sources"] if source["ref"] == ALLOWED_PLACEHOLDER
    ]
    if placeholders:
        print("unresolved upstream pins:", ", ".join(placeholders))
    print(f"validated {len(data['sources'])} upstream sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

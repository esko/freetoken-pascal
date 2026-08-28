from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "upstreams.yaml"
DEFAULT_SCHEMA = ROOT / "manifests" / "upstreams.schema.json"
DEFAULT_NOTICE = ROOT / "NOTICE"
NOTICE_ID_RE = re.compile(r"^- ([a-z0-9][a-z0-9-]*):", re.MULTILINE)


def load_document(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _display_path(parts: list[Any]) -> str:
    if not parts:
        return "$"
    return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts)


def validate_manifest(
    data: Any,
    schema: dict[str, Any],
    *,
    root: Path,
    notice_text: str,
) -> list[str]:
    errors = [
        f"{_display_path(list(error.absolute_path))}: {error.message}"
        for error in sorted(
            Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            ).iter_errors(data),
            key=lambda item: (list(item.absolute_path), item.message),
        )
    ]
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        return errors

    sources = data["sources"]
    ids = [source.get("id") for source in sources if isinstance(source, dict)]
    duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if duplicates:
        errors.append(f"duplicate source ids: {', '.join(duplicates)}")

    notice_ids = set(NOTICE_ID_RE.findall(notice_text))
    expected_notice_ids: set[str] = set()

    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = source.get("id", "<unknown>")
        imports = source.get("imports")
        if source.get("notice_required") is True and isinstance(source_id, str):
            expected_notice_ids.add(source_id)
        if source.get("usage") == "imported" and not imports:
            errors.append(f"{source_id}: imported source must declare at least one import")
        if source.get("usage") != "imported" and imports:
            errors.append(f"{source_id}: only imported sources may declare destination paths")
        if not isinstance(imports, list):
            continue
        for index, imported in enumerate(imports):
            if not isinstance(imported, dict):
                continue
            if imported.get("source_ref") != source.get("ref"):
                errors.append(f"{source_id}.imports[{index}]: source_ref must equal source ref")
            if imported.get("license") != source.get("license"):
                errors.append(
                    f"{source_id}.imports[{index}]: import license must equal source license"
                )
            destination = imported.get("destination_path")
            if not isinstance(destination, str):
                continue
            destination_path = (root / destination).resolve()
            try:
                destination_path.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{source_id}.imports[{index}]: destination escapes repository root")
                continue
            if not destination_path.exists():
                errors.append(
                    f"{source_id}.imports[{index}]: destination does not exist: {destination}"
                )

    missing_notice = sorted(expected_notice_ids - notice_ids)
    unexpected_notice = sorted(notice_ids - expected_notice_ids)
    if missing_notice:
        errors.append(f"NOTICE missing required source ids: {', '.join(missing_notice)}")
    if unexpected_notice:
        errors.append(f"NOTICE lists non-imported source ids: {', '.join(unexpected_notice)}")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the upstream provenance ledger")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--notice", type=Path, default=DEFAULT_NOTICE)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data = load_document(args.manifest)
        schema = load_document(args.schema)
        notice_text = args.notice.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        print(f"ERROR: unable to read provenance inputs: {error}", file=sys.stderr)
        return 1

    errors = validate_manifest(data, schema, root=args.root, notice_text=notice_text)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1

    imported = sum(source["usage"] == "imported" for source in data["sources"])
    print(f"validated {len(data['sources'])} pinned upstream sources ({imported} imported)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

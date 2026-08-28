from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = [
    path
    for path in ROOT.rglob("*.md")
    if not {
        "vendor",
        "third_party",
        "upstream",
        ".git",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "build",
    }.intersection(path.parts)
]
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ADR_RE = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")


def local_target(source: Path, raw: str) -> Path | None:
    target = raw.split("#", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    return (source.parent / target).resolve()


def main() -> int:
    errors: list[str] = []

    for path in MARKDOWN:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("#"):
            errors.append(f"{path.relative_to(ROOT)}: must start with a heading")
        for raw in LINK_RE.findall(text):
            target = local_target(path, raw)
            if target is not None and not target.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing local link {raw!r}")

    adr_dir = ROOT / "docs" / "adr"
    numbers: list[int] = []
    if adr_dir.exists():
        for path in adr_dir.glob("[0-9][0-9][0-9][0-9]-*.md"):
            match = ADR_RE.match(path.name)
            if not match:
                errors.append(f"{path.relative_to(ROOT)}: invalid ADR filename")
                continue
            numbers.append(int(match.group(1)))
            text = path.read_text(encoding="utf-8")
            for required in ("- Status:", "## Context", "## Decision", "## Consequences"):
                if required not in text:
                    errors.append(f"{path.relative_to(ROOT)}: missing {required!r}")
        if numbers and sorted(numbers) != list(range(min(numbers), max(numbers) + 1)):
            errors.append(f"ADR sequence has gaps: {sorted(numbers)}")

    required_docs = {
        "README.md",
        "AGENTS.md",
        "docs/product-scope.md",
        "docs/architecture.md",
        "docs/implementation-plan.md",
        "docs/orchestrator-guide.md",
        "docs/testing-strategy.md",
        "docs/release-criteria.md",
    }
    for rel in sorted(required_docs):
        if not (ROOT / rel).exists():
            errors.append(f"missing required document: {rel}")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"validated {len(MARKDOWN)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check_upstream_manifest.py"
REPORTER = ROOT / "scripts" / "report_upstream_changes.py"
SCHEMA = ROOT / "manifests" / "upstreams.schema.json"


def valid_manifest() -> dict:
    return {
        "schema_version": 2,
        "updated": "2026-08-28",
        "policy": {
            "require_commit_sha": True,
            "require_license": True,
            "require_responsible_issue": True,
            "moving_refs_are_prohibited_in_release": True,
        },
        "sources": [
            {
                "id": "source-one",
                "repository": "https://github.com/example/source-one",
                "upstream_ref": "refs/heads/main",
                "ref": "1" * 40,
                "license": "Apache-2.0",
                "role": "test source",
                "usage": "imported",
                "responsible_issue": 6,
                "notice_required": True,
                "imports": [
                    {
                        "source_path": ".",
                        "destination_path": ".",
                        "source_ref": "1" * 40,
                        "method": "merge",
                        "local_modifications": [],
                        "license": "Apache-2.0",
                        "header_policy": "preserved",
                        "responsible_issue": 6,
                    }
                ],
            }
        ],
    }


def run_checker(tmp_path: Path, data: dict, notice: str = "- source-one: test\n"):
    manifest = tmp_path / "upstreams.yaml"
    notice_path = tmp_path / "NOTICE"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    notice_path.write_text(notice, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--manifest",
            str(manifest),
            "--schema",
            str(SCHEMA),
            "--notice",
            str(notice_path),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )


def test_valid_manifest_passes(tmp_path: Path):
    result = run_checker(tmp_path, valid_manifest())

    assert result.returncode == 0
    assert "validated 1 pinned upstream sources (1 imported)" in result.stdout


def test_moving_ref_is_rejected(tmp_path: Path):
    data = valid_manifest()
    data["sources"][0]["ref"] = "main"

    result = run_checker(tmp_path, data)

    assert result.returncode == 1
    assert "does not match '^[0-9a-f]{40}$'" in result.stderr


def test_duplicate_ids_are_rejected(tmp_path: Path):
    data = valid_manifest()
    duplicate = copy.deepcopy(data["sources"][0])
    duplicate["usage"] = "reference"
    duplicate["notice_required"] = False
    duplicate["imports"] = []
    data["sources"].append(duplicate)

    result = run_checker(tmp_path, data)

    assert result.returncode == 1
    assert "duplicate source ids: source-one" in result.stderr


def test_missing_license_is_rejected(tmp_path: Path):
    data = valid_manifest()
    del data["sources"][0]["license"]

    result = run_checker(tmp_path, data)

    assert result.returncode == 1
    assert "'license' is a required property" in result.stderr


def test_imported_source_requires_path_ledger(tmp_path: Path):
    data = valid_manifest()
    data["sources"][0]["imports"] = []

    result = run_checker(tmp_path, data)

    assert result.returncode == 1
    assert "imported source must declare at least one import" in result.stderr


def test_missing_import_destination_is_rejected(tmp_path: Path):
    data = valid_manifest()
    data["sources"][0]["imports"][0]["destination_path"] = "missing/file.py"

    result = run_checker(tmp_path, data)

    assert result.returncode == 1
    assert "destination does not exist: missing/file.py" in result.stderr


def test_notice_and_manifest_must_agree(tmp_path: Path):
    result = run_checker(tmp_path, valid_manifest(), notice="- another-source: test\n")

    assert result.returncode == 1
    assert "NOTICE missing required source ids: source-one" in result.stderr
    assert "NOTICE lists non-imported source ids: another-source" in result.stderr


def test_repository_manifest_and_offline_report_are_machine_readable():
    check = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = subprocess.run(
        [sys.executable, str(REPORTER), "--offline", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert check.returncode == 0, check.stderr
    assert report.returncode == 0, report.stderr
    payload = json.loads(report.stdout)
    assert len(payload["sources"]) == 12
    assert {row["status"] for row in payload["sources"]} == {"offline"}
    assert all(len(row["pinned"]) == 40 for row in payload["sources"])

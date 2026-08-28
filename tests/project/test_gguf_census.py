from __future__ import annotations

import json
from pathlib import Path

import pytest
from freetoken.gguf_census import build_quant_census, model_sha256

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"


def test_census_matches_schema_fixture() -> None:
    actual = build_quant_census(FIXTURES / "gguf/valid-heterogeneous.gguf")
    expected = json.loads(
        (FIXTURES / "results/quant-census.json").read_text(encoding="utf-8")
    )

    assert actual == expected


def test_declared_identity_is_never_reported_as_measured() -> None:
    source = FIXTURES / "gguf/valid-heterogeneous.gguf"
    declared = {
        source.name: {
            "size": source.stat().st_size,
            "sha256": "1" * 64,
        }
    }

    census = build_quant_census(
        source,
        declared_shards=declared,
        verify_sha256=False,
    )

    assert census["evidence_status"] == "artifact-metadata"
    assert census["shards"][0]["sha256_status"] == "declared"
    assert census["model_sha256"] == "1" * 64


def test_declared_identity_mismatch_fails_closed() -> None:
    source = FIXTURES / "gguf/valid-heterogeneous.gguf"
    declared = {source.name: {"size": source.stat().st_size, "sha256": "1" * 64}}

    with pytest.raises(ValueError, match="does not match declared"):
        build_quant_census(source, declared_shards=declared, verify_sha256=True)


def test_multi_shard_model_identity_is_ordered_and_unambiguous() -> None:
    shards = [
        {"name": "a.gguf", "size": 10, "sha256": "1" * 64},
        {"name": "b.gguf", "size": 20, "sha256": "2" * 64},
    ]

    expected = "950fa5f999b5ba8ff003782f537f50f92b40106af172319050479f0ff21e3379"
    assert model_sha256(shards) == expected
    assert model_sha256(list(reversed(shards))) != model_sha256(shards)


def test_pinned_qwen_variants_are_complete_and_quant_counts_sum() -> None:
    manifest = json.loads((ROOT / "manifests/qwen38-gguf.json").read_text(encoding="utf-8"))

    assert len(manifest["revision"]) == 40
    assert set(manifest["variants"]) == {"UD-Q3_K_XL", "UD-Q4_K_XL"}
    for variant in manifest["variants"].values():
        assert sum(variant["quant_type_counts"].values()) == variant["tensor_count"]
        assert all(len(shard["sha256"]) == 64 for shard in variant["shards"])


@pytest.mark.parametrize(
    ("variant", "filename"),
    [
        ("UD-Q3_K_XL", "qwen38-q3-census.metadata.json"),
        ("UD-Q4_K_XL", "qwen38-q4-census.metadata.json"),
    ],
)
def test_checked_in_qwen_metadata_census_matches_pinned_manifest(
    variant: str,
    filename: str,
) -> None:
    manifest = json.loads((ROOT / "manifests/qwen38-gguf.json").read_text(encoding="utf-8"))
    census = json.loads((FIXTURES / "results" / filename).read_text(encoding="utf-8"))
    expected = manifest["variants"][variant]

    assert census["evidence_status"] == "artifact-metadata"
    assert census["tensor_count"] == expected["tensor_count"] == 1224
    assert {key: value["tensors"] for key, value in census["by_quant_type"].items()} == (
        expected["quant_type_counts"]
    )
    assert len(census["expert_layers"]) == 48
    assert len(census["expert_pools"]) == 48 * 3
    assert all(shard["sha256_status"] == "declared" for shard in census["shards"])

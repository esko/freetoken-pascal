from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _config(**overrides):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    values = {
        "model_path": "/tmp/freetoken-host-bank-policy",
        "tp_info": DistributedInfo(rank=0, size=1),
        "dtype": torch.bfloat16,
    }
    values.update(overrides)
    return EngineConfig(**values)


def _write_ftw_index(path, *, nbytes: int = 2048):
    path.mkdir()
    (path / "freetoken_weight.json").write_text(
        json.dumps(
            {
                "format": "freetoken_weight",
                "version": 1,
                "expert_bank_num_layers": 1,
                "tensors": [
                    {
                        "name": "gate_up",
                        "kind": "experts_bank",
                        "dtype": "uint8",
                        "shape": [2, nbytes // 2],
                        "global_off": 0,
                        "nbytes": nbytes,
                    }
                ],
                "shards": [],
            }
        )
    )


def test_engine_config_default_keeps_legacy_no_policy_path():
    config = _config()

    assert config.host_bank_policy is None


def test_engine_config_rejects_non_policy_host_bank_value():
    from freetoken.engine.config import EngineConfig
    from freetoken.distributed import DistributedInfo

    with pytest.raises(TypeError, match="host_bank_policy"):
        EngineConfig(
            model_path="/tmp/freetoken-host-bank-policy",
            tp_info=DistributedInfo(rank=0, size=1),
            dtype=torch.bfloat16,
            host_bank_policy=object(),
        )


def test_engine_config_rejects_unbounded_pinned_policy():
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    from freetoken.moe.host_banks import HostBankPolicy

    with pytest.raises(ValueError, match="finite max_pinned_bytes"):
        EngineConfig(
            model_path="/tmp/freetoken-host-bank-policy",
            tp_info=DistributedInfo(rank=0, size=1),
            dtype=torch.bfloat16,
            host_bank_policy=HostBankPolicy(strategy="pinned", max_pinned_bytes=None),
        )


def test_mutated_policy_is_revalidated_before_ftw_preflight(tmp_path):
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "ftw"
    _write_ftw_index(checkpoint)
    policy = HostBankPolicy(strategy="pageable")
    policy.strategy = "pinned"
    policy.max_pinned_bytes = None

    with pytest.raises(ValueError, match="finite max_pinned_bytes"):
        policy.prepare_layer_bytes([4096])


def test_ftw_policy_preflight_runs_before_loader_and_reports_page_rounded_bytes(tmp_path):
    from freetoken.moe.expert_banks import ExpertBanks, load_expert_banks
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "ftw"
    _write_ftw_index(checkpoint, nbytes=2048)
    policy = HostBankPolicy(strategy="pageable")
    seen = {}

    def fake_load(path, **kwargs):
        seen["prepared"] = policy.accounting.as_dict()
        seen["policy"] = kwargs["host_bank_policy"]
        return ExpertBanks(
            "q4_0",
            {"gate_up": []},
            layer_residency=["pageable"],
            host_bank_accounting=policy.accounting.as_dict(),
        )

    config = SimpleNamespace(num_moe_layers=1)
    with patch("freetoken.checkpoint.ftw.load_ftw_banks", fake_load):
        result = load_expert_banks(
            str(checkpoint),
            config,
            device=torch.device("cpu"),
            dtype=torch.uint8,
            host_bank_policy=policy,
        )

    assert seen["policy"] is policy
    assert seen["prepared"]["source_bytes"] == 4096
    assert seen["prepared"]["strategy"] == "pageable"
    assert result.host_bank_accounting["source_bytes"] == 4096


def test_ftw_policy_budget_rejects_before_loader_is_called(tmp_path):
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "ftw"
    _write_ftw_index(checkpoint, nbytes=8192)
    policy = HostBankPolicy(strategy="pinned", max_pinned_bytes=4096)
    config = SimpleNamespace(num_moe_layers=1)

    with patch("freetoken.checkpoint.ftw.load_ftw_banks") as load:
        with pytest.raises(ValueError, match="pinned host-bank budget"):
            load_expert_banks(
                str(checkpoint),
                config,
                device=torch.device("cpu"),
                dtype=torch.uint8,
                host_bank_policy=policy,
            )

    load.assert_not_called()


def test_explicit_policy_rejects_non_ftw_provider_before_read():
    from freetoken.moe.expert_banks import load_expert_banks
    from freetoken.moe.host_banks import HostBankPolicy

    with pytest.raises(ValueError, match="FTW"):
        load_expert_banks(
            "/tmp/not-an-ftw-checkpoint",
            SimpleNamespace(num_moe_layers=1),
            device=torch.device("cpu"),
            dtype=torch.uint8,
            host_bank_policy=HostBankPolicy(strategy="pageable"),
        )

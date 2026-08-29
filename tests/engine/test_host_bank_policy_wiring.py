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
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

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


def test_pinned_policy_rejects_skip_pin_hook(monkeypatch):
    from freetoken.moe.host_banks import HostBankPolicy

    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    with pytest.raises(ValueError, match="SKIP_BANK_PIN"):
        HostBankPolicy(strategy="pinned", max_pinned_bytes=4096).prepare_layer_bytes([4096])


@pytest.mark.parametrize(
    ("strategy", "message"),
    (
        ("pageable", "pageable.*preflight-only"),
        ("bounded-staging", "bounded-staging.*preflight-only"),
    ),
)
def test_engine_rejects_unwired_nonpinned_strategies(strategy, message):
    from freetoken.engine.engine import Engine
    from freetoken.moe.host_banks import HostBankPolicy

    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace()
    config = SimpleNamespace(
        host_bank_policy=HostBankPolicy(
            strategy=strategy,
            staging_bytes=4096 if strategy == "bounded-staging" else 0,
            max_staging_bytes=4096 if strategy == "bounded-staging" else 0,
        ),
        model_config=SimpleNamespace(num_moe_layers=1),
        moe_cache_auto=False,
    )
    with patch("freetoken.engine.engine._guard_qwen_gguf_engine_setup"):
        with pytest.raises(NotImplementedError, match=message):
            engine._init_offload_moe_cache(config)


def test_engine_cleanup_drops_cache_references_before_bank_owners():
    from freetoken.engine.engine import Engine

    events = []
    engine = Engine.__new__(Engine)
    engine.cpu_moe_executor = object()
    engine.moe_offload_cache = SimpleNamespace(
        cpu_executor=object(),
        bank_sources={"gate_up": [object()]},
        banks=[object()],
        bank_caches={"gate_up": object()},
        gate_up_alpha=object(),
        down_alpha=object(),
        _copy_src_ptrs=object(),
        _copy_dst_ptrs=object(),
        _copy_feat_bytes=object(),
    )

    class Banks:
        def close(self):
            events.append((not engine.moe_offload_cache.bank_sources, engine.cpu_moe_executor))

    engine._expert_banks = Banks()
    engine._cleanup_host_bank_resources()

    assert events == [(True, None)]
    assert engine._expert_banks is None


def test_engine_rejects_selective_pinned_policy():
    from freetoken.engine.engine import Engine
    from freetoken.moe.host_banks import HostBankPolicy

    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace()
    config = SimpleNamespace(
        host_bank_policy=HostBankPolicy(
            strategy="pinned", max_pinned_bytes=4096, selected_layers=(0,)
        ),
        model_config=SimpleNamespace(num_moe_layers=1),
        moe_cache_auto=False,
    )
    with patch("freetoken.engine.engine._guard_qwen_gguf_engine_setup"):
        with pytest.raises(NotImplementedError, match="selected host-bank layers"):
            engine._init_offload_moe_cache(config)


def test_engine_constructor_rolls_back_late_startup_failure():
    from freetoken.engine.engine import Engine

    cleanup = []

    def fail(_self, _config):
        raise RuntimeError("late startup failure")

    def rollback(_self):
        cleanup.append(True)

    with patch.object(Engine, "_initialize", fail), patch.object(
        Engine, "_cleanup_host_bank_resources", rollback
    ):
        with pytest.raises(RuntimeError, match="late startup failure"):
            Engine(object())

    assert cleanup == [True]


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


def test_ftw_no_swap_preflight_runs_before_index_open(tmp_path):
    from freetoken.checkpoint.ftw import prepare_ftw_host_bank_policy
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "ftw"
    _write_ftw_index(checkpoint)
    policy = HostBankPolicy(
        strategy="pageable",
        require_no_swap=True,
        swap_probe_reader=lambda path: {
            "/proc/self/status": "VmSwap: 0 kB\n",
            "/proc/meminfo": "SwapTotal: 1 MiB\nSwapFree: 1 MiB\n",
            "/proc/swaps": "Filename Type Size Used Priority\n",
        }[path],
    )

    with patch("builtins.open", side_effect=AssertionError("index opened too early")):
        with pytest.raises(ValueError, match=r"require_no_swap.*swap-active"):
            prepare_ftw_host_bank_policy(str(checkpoint), num_layers=1, policy=policy)


def test_ftw_no_swap_snapshot_is_threaded_without_second_probe(tmp_path):
    from freetoken.moe.expert_banks import ExpertBanks, load_expert_banks
    from freetoken.moe.host_banks import HostBankPolicy

    checkpoint = tmp_path / "ftw"
    _write_ftw_index(checkpoint)
    reads = []
    files = {
        "/proc/self/status": "VmSwap: 0 kB\n",
        "/proc/meminfo": "SwapTotal: 0 kB\nSwapFree: 0 kB\n",
        "/proc/swaps": "Filename Type Size Used Priority\n",
    }

    def read(path):
        reads.append(path)
        return files[path]

    policy = HostBankPolicy(
        strategy="pageable", require_no_swap=True, swap_probe_reader=read
    )
    seen = {}

    def fake_load(path, **kwargs):
        seen["swap_probe"] = kwargs["swap_probe"]
        return ExpertBanks(
            "q4_0",
            {"gate_up": []},
            layer_residency=["pageable"],
            host_bank_accounting=policy.accounting.as_dict(),
        )

    with patch("freetoken.checkpoint.ftw.load_ftw_banks", fake_load):
        load_expert_banks(
            str(checkpoint),
            SimpleNamespace(num_moe_layers=1),
            device=torch.device("cpu"),
            dtype=torch.uint8,
            host_bank_policy=policy,
        )

    assert len(reads) == 3
    assert seen["swap_probe"] is policy.swap_probe


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

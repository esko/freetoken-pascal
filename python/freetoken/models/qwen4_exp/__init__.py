"""Text-only Qwen3.8-Flash-Next (``qwen4_exp``) model package.

The runtime uses the upstream modular attention/GDN/hyper-connection/MoE/PLE operators. The
GGUF parser and explicit host CPU expert attachment remain downstream-only seams for the
Pascal reference path and are never selected implicitly by ordinary HF model loading.
"""

from .config import parse_config, parse_gguf_config
from .gguf import iter_gguf_weights
from .model import Qwen4ExpForCausalLM
from .weight import (
    iter_weights,
    iter_weights_parallel,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    load_ple_table,
)


def setup_offload_expert_banks(model_path, model_config, **kwargs):
    """Dispatch native quantized banks and fail closed for heterogeneous GGUF experts.

    The generic qwen3.5 provider handles the upstream FP8/NVFP4 bank layouts. GGUF experts are
    deliberately kept on the explicit ``QwenGGUFCpuExpertBundle`` seam until a matching low-bit
    executor is registered, so they are never silently reinterpreted as NVFP4.
    """
    if getattr(model_config, "expert_quant", "none") != "gguf":
        from freetoken.models.qwen3_5_moe.weight import setup_offload_expert_banks as setup_native

        return setup_native(model_path, model_config, **kwargs)

    from freetoken.gguf_host import inspect_qwen_host_layout
    from freetoken.gguf_types import MOE_VEC_TYPES

    layout = inspect_qwen_host_layout(model_path, supported_expert_types=MOE_VEC_TYPES).experts
    ledger = ", ".join(
        f"pool{pool.pool_id}:{pool.projection}/{pool.quant_name}/"
        f"{pool.output_dim}x{pool.input_dim}/layers={list(pool.layers)}"
        for pool in layout.slot_pools
    )
    raise NotImplementedError(
        "Qwen3.8 GGUF heterogeneous expert sources validated, but no low-bit expert "
        f"executor is registered yet ({ledger}); refusing to reinterpret them as NVFP4"
    )


__all__ = [
    "Qwen4ExpForCausalLM",
    "iter_gguf_weights",
    "iter_weights",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
    "parse_config",
    "parse_gguf_config",
    "setup_offload_expert_banks",
]

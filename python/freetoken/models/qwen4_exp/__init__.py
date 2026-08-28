from .config import parse_config, parse_gguf_config
from .gguf import iter_gguf_weights
from .model import Qwen4ExpForCausalLM
from .weight import (
    iter_weights,
    iter_weights_parallel,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
)
from .weight import setup_offload_expert_banks as _setup_quantized_expert_banks


def setup_offload_expert_banks(model_path, model_config, **kwargs):
    """Dispatch legacy FP8/NVFP4 banks and fail explicitly at the GGUF executor seam."""
    if model_config.expert_quant != "gguf":
        return _setup_quantized_expert_banks(model_path, model_config, **kwargs)

    from freetoken.gguf_host import inspect_qwen_host_layout
    from freetoken.gguf_types import MOE_VEC_TYPES

    layout = inspect_qwen_host_layout(
        model_path,
        supported_expert_types=MOE_VEC_TYPES,
    ).experts
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
    "parse_config",
    "parse_gguf_config",
    "setup_offload_expert_banks",
]

"""Explicit borrowed-bundle attachment for the Qwen4-Exp model graph.

The helper deliberately knows only the model's small structural/configuration
surface. It does not import or construct a Qwen model, Engine, or CUDA operator,
which keeps fake-layer lifecycle tests independent from model construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle
from freetoken.moe.gguf_layer import QwenGGUFCpuMoELayer


@dataclass(frozen=True)
class GGUFCpuExpertAttachment:
    """Reversible state for one borrowed Qwen GGUF expert attachment."""

    bundle: QwenGGUFCpuExpertBundle
    replacements: tuple[tuple[object, object, object, object], ...]


def validate_gguf_cpu_attachment(model: Any, bundle: QwenGGUFCpuExpertBundle) -> tuple[object, ...]:
    """Validate all model/bundle metadata before constructing any adapter."""
    if not isinstance(bundle, QwenGGUFCpuExpertBundle):
        raise TypeError("Qwen4-Exp model attachment requires a QwenGGUFCpuExpertBundle")
    if bundle.closed:
        raise RuntimeError("Qwen GGUF CPU expert bundle is closed")

    config = model._config
    layers = tuple(model.layers.op_list)
    expected_ids = tuple(range(config.num_layers))
    try:
        model_ids = tuple(layer._layer_id for layer in layers)
    except AttributeError as error:
        raise ValueError("Qwen4-Exp model layers must expose _layer_id") from error
    if model_ids != expected_ids:
        raise ValueError(
            f"Qwen4-Exp model layer IDs {model_ids} do not match expected layer IDs {expected_ids}"
        )
    if tuple(bundle.layout.layers) != expected_ids:
        raise ValueError(
            f"Qwen GGUF CPU bundle layer IDs {tuple(bundle.layout.layers)} do not match "
            f"model layer IDs {expected_ids}"
        )
    if bundle.layout.top_k != config.num_experts_per_tok:
        raise ValueError(
            f"Qwen GGUF CPU bundle top_k={bundle.layout.top_k} does not match "
            f"model top_k={config.num_experts_per_tok}"
        )

    for layer_id in expected_ids:
        try:
            gate = bundle.layout.descriptor(layer_id, "gate")
            up = bundle.layout.descriptor(layer_id, "up")
            down = bundle.layout.descriptor(layer_id, "down")
        except Exception as error:
            raise ValueError(
                f"Qwen GGUF CPU bundle is missing projections for layer {layer_id}"
            ) from error
        for descriptor in (gate, up, down):
            if descriptor.num_experts != config.num_experts:
                raise ValueError(
                    f"Qwen GGUF CPU bundle layer {layer_id} {descriptor.projection} "
                    f"num_experts={descriptor.num_experts} does not match model "
                    f"num_experts={config.num_experts}"
                )
        if (gate.input_dim, gate.output_dim) != (
            config.hidden_size,
            config.moe_intermediate_size,
        ):
            raise ValueError(
                f"Qwen GGUF CPU bundle layer {layer_id} gate geometry "
                f"({gate.input_dim}, {gate.output_dim}) does not match model geometry "
                f"({config.hidden_size}, {config.moe_intermediate_size})"
            )
        if (up.input_dim, up.output_dim) != (
            config.hidden_size,
            config.moe_intermediate_size,
        ):
            raise ValueError(
                f"Qwen GGUF CPU bundle layer {layer_id} up geometry "
                f"({up.input_dim}, {up.output_dim}) does not match model geometry "
                f"({config.hidden_size}, {config.moe_intermediate_size})"
            )
        if (down.input_dim, down.output_dim) != (
            config.moe_intermediate_size,
            config.hidden_size,
        ):
            raise ValueError(
                f"Qwen GGUF CPU bundle layer {layer_id} down geometry "
                f"({down.input_dim}, {down.output_dim}) does not match model geometry "
                f"({config.moe_intermediate_size}, {config.hidden_size})"
            )

    try:
        from freetoken.distributed.info import try_get_tp_info

        tp_info = try_get_tp_info()
    except ImportError:  # pragma: no cover - distributed package ships with runtime
        tp_info = None
    if tp_info is not None and tp_info.size != 1:
        raise ValueError(
            f"Qwen GGUF CPU model attachment requires TP=1, got tensor-parallel size {tp_info.size}"
        )

    executor = bundle.executor
    executor_reference = getattr(executor, "_reference", None)
    activation = getattr(
        executor,
        "activation",
        getattr(executor_reference, "activation", None),
    )
    if activation != config.hidden_act:
        raise ValueError(
            f"Qwen GGUF CPU bundle activation={activation!r} does not match "
            f"model activation={config.hidden_act!r}"
        )
    router_weight = getattr(
        executor,
        "apply_router_weight_on_input",
        getattr(executor_reference, "apply_router_weight_on_input", None),
    )
    if router_weight is not False:
        raise ValueError(
            "Qwen GGUF CPU model attachment requires apply_router_weight_on_input=False"
        )
    return layers


def attach_gguf_cpu_expert_bundle(model: Any, bundle: QwenGGUFCpuExpertBundle) -> None:
    """Validate, construct, and atomically install all layer adapters."""
    if getattr(model, "_gguf_cpu_attachment", None) is not None:
        raise RuntimeError("Qwen GGUF CPU expert bundle is already attached")
    layers = validate_gguf_cpu_attachment(model, bundle)
    config = model._config

    layer_records: list[tuple[object, object, object]] = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "experts"):
            raise ValueError(
                f"Qwen4-Exp layer {getattr(layer, '_layer_id', '?')} has no sparse experts"
            )
        layer_records.append((layer, mlp, mlp.experts))

    replacements: list[tuple[object, object, object, object]] = []
    for layer, mlp, original in layer_records:
        adapter = QwenGGUFCpuMoELayer(
            bundle,
            layer_id=layer._layer_id,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=bool(config.norm_topk_prob),
            activation=config.hidden_act,
            apply_router_weight_on_input=False,
        )
        replacements.append((layer, mlp, original, adapter))

    attachment = GGUFCpuExpertAttachment(bundle, tuple(replacements))
    try:
        for _layer, mlp, _original, adapter in replacements:
            mlp.experts = adapter
    except BaseException:
        for _layer, mlp, original, _adapter in reversed(replacements):
            mlp.experts = original
        raise
    model._gguf_cpu_attachment = attachment


def detach_gguf_cpu_expert_bundle(model: Any) -> None:
    """Restore exact pre-attachment objects; never close the borrowed bundle."""
    attachment = model._gguf_cpu_attachment
    if attachment is None:
        return

    for layer, mlp, _original, adapter in attachment.replacements:
        if getattr(layer, "mlp", None) is not mlp or getattr(mlp, "experts", None) is not adapter:
            raise RuntimeError("Qwen GGUF CPU expert attachment was modified outside its lifecycle")

    try:
        for _layer, mlp, original, _adapter in attachment.replacements:
            mlp.experts = original
    except BaseException:
        for _layer, mlp, _original, adapter in reversed(attachment.replacements):
            mlp.experts = adapter
        raise
    model._gguf_cpu_attachment = None


def append_original_expert_state(
    model: Any,
    result: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Keep runtime adapter replacement out of the model state-dict surface."""
    attachment = model._gguf_cpu_attachment
    if attachment is None:
        return result
    for index, (_layer, _mlp, original, _adapter) in enumerate(attachment.replacements):
        original_state_dict = getattr(original, "state_dict", None)
        if original_state_dict is None:
            continue
        expert_prefix = (
            f"{prefix}.layers.{index}.mlp.experts" if prefix else f"layers.{index}.mlp.experts"
        )
        original_state_dict(prefix=expert_prefix, result=result)
    return result


__all__ = [
    "GGUFCpuExpertAttachment",
    "append_original_expert_state",
    "attach_gguf_cpu_expert_bundle",
    "detach_gguf_cpu_expert_bundle",
    "validate_gguf_cpu_attachment",
]

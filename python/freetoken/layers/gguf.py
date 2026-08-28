"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized inside the borrowed llama.cpp kernels or an explicit reference fallback.

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch

from freetoken.gguf_types import (
    BLOCK_SHAPE,
    DEQUANT_TYPES,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    GGML_UNQUANTIZED,
    MMQ_TYPES,
    MMVQ_TYPES,
    row_bytes,
)

from .base import BaseOP

_UNQUANTIZED_DTYPE = {
    GGML_F32: torch.float32,
    GGML_F16: torch.float16,
    GGML_BF16: torch.bfloat16,
}

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in GGML_UNQUANTIZED:
        weight = qweight
        if weight.dtype == torch.uint8:
            weight = weight.view(_UNQUANTIZED_DTYPE[qweight_type])
        return (x.to(weight.dtype) @ weight.T).to(x.dtype)
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in MMVQ_TYPES:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in MMQ_TYPES:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in DEQUANT_TYPES:
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(
            out_features,
            row_bytes(in_features, quant_type),
            dtype=torch.uint8,
        )
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out


class GGUFLMHead(GGUFLinear):
    """Untied GGUF output projection with prefill last-token slicing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return super().forward(x)


class GGUFInputPermutedLinear(GGUFLinear):
    """Packed linear that presents activations in the converter's column order."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        input_permutation: torch.Tensor,
        has_bias: bool = False,
    ):
        super().__init__(in_features, out_features, quant_type, has_bias=has_bias)
        permutation = input_permutation.to(dtype=torch.long, device="cpu").contiguous()
        if permutation.shape != (in_features,) or set(permutation.tolist()) != set(
            range(in_features)
        ):
            raise ValueError("input_permutation must contain every input index exactly once")
        self._input_permutation = permutation
        self._device_permutation: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._device_permutation is None
            or self._device_permutation.device != x.device
        ):
            self._device_permutation = self._input_permutation.to(device=x.device)
        return super().forward(x.index_select(-1, self._device_permutation))


class GGUFMergedLinear(BaseOP):
    """Merged projection whose independently packed parts may use different types."""

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        quant_types: list[int],
        has_bias: bool = False,
    ):
        if len(output_sizes) != len(quant_types):
            raise ValueError(
                f"output_sizes length {len(output_sizes)} != "
                f"quant_types length {len(quant_types)}"
            )
        if not output_sizes or any(output_size <= 0 for output_size in output_sizes):
            raise ValueError(f"all output_sizes must be positive, got {output_sizes}")
        unsupported = [
            quant_type
            for quant_type in quant_types
            if quant_type not in MMVQ_TYPES and quant_type not in GGML_UNQUANTIZED
        ]
        if unsupported:
            names = [GGML_NAME.get(quant_type, str(quant_type)) for quant_type in unsupported]
            raise NotImplementedError(f"unsupported GGUF merged quant types: {names}")

        self.in_features = in_features
        self.output_sizes = list(output_sizes)
        self.out_features = sum(output_sizes)
        self._quant_types = list(quant_types)
        self.part_names: list[str] = []
        for index, (out_features, quant_type) in enumerate(
            zip(output_sizes, quant_types, strict=True)
        ):
            name = f"qweight_{index}"
            self.part_names.append(name)
            setattr(
                self,
                name,
                torch.empty(
                    out_features,
                    row_bytes(in_features, quant_type),
                    dtype=torch.uint8,
                ),
            )
        self.bias = torch.empty(self.out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat(
            [
                fused_mul_mat_gguf(x, getattr(self, name), quant_type)
                for name, quant_type in zip(
                    self.part_names, self._quant_types, strict=True
                )
            ],
            dim=-1,
        )
        return output if self.bias is None else output + self.bias


def gguf_merged_or_plain(
    in_features: int,
    output_sizes: list[int],
    quant_types: list[int],
    has_bias: bool = False,
) -> GGUFLinear | GGUFMergedLinear:
    """Select one packed buffer for a uniform group, or one buffer per mixed part."""
    if not quant_types or len(output_sizes) != len(quant_types):
        raise ValueError("output_sizes and quant_types must have the same nonzero length")
    if len(set(quant_types)) == 1:
        return GGUFLinear(
            in_features,
            sum(output_sizes),
            quant_types[0],
            has_bias=has_bias,
        )
    return GGUFMergedLinear(
        in_features,
        output_sizes,
        quant_types,
        has_bias=has_bias,
    )


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        if self._quant_type in GGML_UNQUANTIZED:
            y = rows.view(_UNQUANTIZED_DTYPE[self._quant_type]).to(torch.bfloat16)
        else:
            y = ggml_dequantize(
                rows,
                self._quant_type,
                flat.shape[0],
                self.embedding_dim,
                torch.bfloat16,
            )
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale,
                    dtype=y.dtype,
                    device=y.device,
                )
            y = y * self._embed_scale_t
        return y


__all__ = [
    "GGUFEmbedding",
    "GGUFInputPermutedLinear",
    "GGUFLMHead",
    "GGUFLinear",
    "GGUFMergedLinear",
    "fused_mul_mat_gguf",
    "gguf_merged_or_plain",
]

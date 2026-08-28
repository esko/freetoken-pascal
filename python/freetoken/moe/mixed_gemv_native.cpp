// Baseline dispatch for the Issue #16 Q5_K/Q5_1/Q8_0 companion primitives.
// Baseline functions use a per-function ISA fence so host -march settings cannot
// add AVX instructions.  Link this translation unit with scalar and AVX2 sources.

#include "mixed_gemv_native.h"

#include <cstddef>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#define FREETOKEN_EXPORT __attribute__((visibility("default")))
#define FREETOKEN_BASELINE_TARGET __attribute__((target("no-avx,no-avx2,no-fma")))
#else
#define FREETOKEN_EXPORT
#define FREETOKEN_BASELINE_TARGET
#endif

namespace {

template <typename Dot>
FREETOKEN_BASELINE_TARGET int gemv_scalar(const uint8_t* rows, int row_count,
                                          int blocks_per_row, int row_stride_bytes,
                                          int block_bytes, int block_elements,
                                          const float* input, float* output, Dot dot) {
  if (rows == nullptr || input == nullptr || output == nullptr || row_count <= 0 ||
      blocks_per_row <= 0 || blocks_per_row > 0x7FFFFFFF / block_bytes ||
      row_stride_bytes != blocks_per_row * block_bytes) {
    return -1;
  }
  for (int row = 0; row < row_count; ++row) {
    const uint8_t* packed_row = rows + static_cast<size_t>(row) * row_stride_bytes;
    float result = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      result += dot(packed_row + static_cast<size_t>(block) * block_bytes,
                    input + static_cast<size_t>(block) * block_elements);
    }
    output[row] = result;
  }
  return 0;
}

FREETOKEN_BASELINE_TARGET bool use_avx2() {
  return freetoken_mixed_cpu_supports_avx2() != 0;
}

}  // namespace

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT int freetoken_mixed_cpu_supports_avx2() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return 0;
#endif
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT float
freetoken_mixed_q5_1_dot(const uint8_t* block, const float* input) {
  return use_avx2() ? freetoken_mixed_q5_1_dot_avx2_impl(block, input)
                    : freetoken_mixed_q5_1_dot_scalar(block, input);
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT void
freetoken_mixed_q5_1_decode(const uint8_t* block, float* output) {
  if (use_avx2()) {
    freetoken_mixed_q5_1_decode_avx2_impl(block, output);
  } else {
    freetoken_mixed_q5_1_decode_scalar(block, output);
  }
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT int freetoken_mixed_q5_1_gemv(
    const uint8_t* rows, int row_count, int blocks_per_row, int row_stride_bytes,
    const float* input, float* output) {
  if (use_avx2()) {
    return freetoken_mixed_q5_1_gemv_avx2_impl(rows, row_count, blocks_per_row,
                                               row_stride_bytes, input, output);
  }
  return gemv_scalar(rows, row_count, blocks_per_row, row_stride_bytes, 24, 32, input, output,
                     freetoken_mixed_q5_1_dot_scalar);
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT float
freetoken_mixed_q8_0_dot(const uint8_t* block, const float* input) {
  return use_avx2() ? freetoken_mixed_q8_0_dot_avx2_impl(block, input)
                    : freetoken_mixed_q8_0_dot_scalar(block, input);
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT void
freetoken_mixed_q8_0_decode(const uint8_t* block, float* output) {
  if (use_avx2()) {
    freetoken_mixed_q8_0_decode_avx2_impl(block, output);
  } else {
    freetoken_mixed_q8_0_decode_scalar(block, output);
  }
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT int freetoken_mixed_q8_0_gemv(
    const uint8_t* rows, int row_count, int blocks_per_row, int row_stride_bytes,
    const float* input, float* output) {
  if (use_avx2()) {
    return freetoken_mixed_q8_0_gemv_avx2_impl(rows, row_count, blocks_per_row,
                                               row_stride_bytes, input, output);
  }
  return gemv_scalar(rows, row_count, blocks_per_row, row_stride_bytes, 34, 32, input, output,
                     freetoken_mixed_q8_0_dot_scalar);
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT float
freetoken_mixed_q5_k_dot(const uint8_t* block, const float* input) {
  return use_avx2() ? freetoken_mixed_q5_k_dot_avx2_impl(block, input)
                    : freetoken_mixed_q5_k_dot_scalar(block, input);
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT void
freetoken_mixed_q5_k_decode(const uint8_t* block, float* output) {
  if (use_avx2()) {
    freetoken_mixed_q5_k_decode_avx2_impl(block, output);
  } else {
    freetoken_mixed_q5_k_decode_scalar(block, output);
  }
}

extern "C" FREETOKEN_BASELINE_TARGET FREETOKEN_EXPORT int freetoken_mixed_q5_k_gemv(
    const uint8_t* rows, int row_count, int blocks_per_row, int row_stride_bytes,
    const float* input, float* output) {
  if (use_avx2()) {
    return freetoken_mixed_q5_k_gemv_avx2_impl(rows, row_count, blocks_per_row,
                                               row_stride_bytes, input, output);
  }
  return gemv_scalar(rows, row_count, blocks_per_row, row_stride_bytes, 176, 256, input, output,
                     freetoken_mixed_q5_k_dot_scalar);
}

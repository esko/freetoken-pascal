// Baseline dispatch for the Issue #16 Q4_K primitive.
// Compile without AVX flags and link with q4_k_scalar.cpp plus q4_k_avx2.cpp.

#include "q4_k_native.h"

#include <cstddef>

#if defined(__GNUC__) || defined(__clang__)
#define FREETOKEN_EXPORT __attribute__((visibility("default")))
#else
#define FREETOKEN_EXPORT
#endif

extern "C" FREETOKEN_EXPORT int freetoken_q4k_cpu_supports_avx2() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return 0;
#endif
}

extern "C" FREETOKEN_EXPORT float freetoken_q4k_dot_avx2(const uint8_t* block,
                                                            const float* input) {
  if (freetoken_q4k_cpu_supports_avx2()) {
    return freetoken_q4k_dot_avx2_impl(block, input);
  }
  return freetoken_q4k_dot_scalar(block, input);
}

extern "C" FREETOKEN_EXPORT void freetoken_q4k_decode_avx2(const uint8_t* block,
                                                             float* output) {
  if (freetoken_q4k_cpu_supports_avx2()) {
    freetoken_q4k_decode_avx2_impl(block, output);
  } else {
    freetoken_q4k_decode_scalar(block, output);
  }
}

extern "C" FREETOKEN_EXPORT void freetoken_q4k_gemv_avx2(
    const uint8_t* rows, int row_count, int blocks_per_row, int row_stride_bytes,
    const float* input, float* output) {
  if (freetoken_q4k_cpu_supports_avx2()) {
    freetoken_q4k_gemv_avx2_impl(rows, row_count, blocks_per_row, row_stride_bytes, input,
                                 output);
    return;
  }
  for (int row = 0; row < row_count; ++row) {
    const uint8_t* packed_row = rows + static_cast<size_t>(row) * row_stride_bytes;
    float result = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      result += freetoken_q4k_dot_scalar(
          packed_row + static_cast<size_t>(block) * 144,
          input + static_cast<size_t>(block) * 256);
    }
    output[row] = result;
  }
}

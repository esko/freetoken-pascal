#pragma once

#include <cstdint>

extern "C" {

int freetoken_q4k_cpu_supports_avx2();
float freetoken_q4k_dot_scalar(const uint8_t* block, const float* input);
void freetoken_q4k_decode_scalar(const uint8_t* block, float* output);
float freetoken_q4k_dot_avx2_impl(const uint8_t* block, const float* input);
void freetoken_q4k_decode_avx2_impl(const uint8_t* block, float* output);
void freetoken_q4k_gemv_avx2_impl(const uint8_t* rows, int row_count,
                                  int blocks_per_row, int row_stride_bytes,
                                  const float* input, float* output);

}

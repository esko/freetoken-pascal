#pragma once

#include <cstdint>

extern "C" {

int freetoken_mixed_cpu_supports_avx2();

float freetoken_mixed_q5_1_dot_scalar(const uint8_t* block, const float* input);
void freetoken_mixed_q5_1_decode_scalar(const uint8_t* block, float* output);
float freetoken_mixed_q5_1_dot_avx2_impl(const uint8_t* block, const float* input);
void freetoken_mixed_q5_1_decode_avx2_impl(const uint8_t* block, float* output);
int freetoken_mixed_q5_1_gemv_avx2_impl(const uint8_t* rows, int row_count,
                                        int blocks_per_row, int row_stride_bytes,
                                        const float* input, float* output);

float freetoken_mixed_q8_0_dot_scalar(const uint8_t* block, const float* input);
void freetoken_mixed_q8_0_decode_scalar(const uint8_t* block, float* output);
float freetoken_mixed_q8_0_dot_avx2_impl(const uint8_t* block, const float* input);
void freetoken_mixed_q8_0_decode_avx2_impl(const uint8_t* block, float* output);
int freetoken_mixed_q8_0_gemv_avx2_impl(const uint8_t* rows, int row_count,
                                        int blocks_per_row, int row_stride_bytes,
                                        const float* input, float* output);

float freetoken_mixed_q5_k_dot_scalar(const uint8_t* block, const float* input);
void freetoken_mixed_q5_k_decode_scalar(const uint8_t* block, float* output);
float freetoken_mixed_q5_k_dot_avx2_impl(const uint8_t* block, const float* input);
void freetoken_mixed_q5_k_decode_avx2_impl(const uint8_t* block, float* output);
int freetoken_mixed_q5_k_gemv_avx2_impl(const uint8_t* rows, int row_count,
                                        int blocks_per_row, int row_stride_bytes,
                                        const float* input, float* output);

float freetoken_mixed_q5_1_dot(const uint8_t* block, const float* input);
void freetoken_mixed_q5_1_decode(const uint8_t* block, float* output);
int freetoken_mixed_q5_1_gemv(const uint8_t* rows, int row_count, int blocks_per_row,
                              int row_stride_bytes, const float* input, float* output);

float freetoken_mixed_q8_0_dot(const uint8_t* block, const float* input);
void freetoken_mixed_q8_0_decode(const uint8_t* block, float* output);
int freetoken_mixed_q8_0_gemv(const uint8_t* rows, int row_count, int blocks_per_row,
                              int row_stride_bytes, const float* input, float* output);

float freetoken_mixed_q5_k_dot(const uint8_t* block, const float* input);
void freetoken_mixed_q5_k_decode(const uint8_t* block, float* output);
int freetoken_mixed_q5_k_gemv(const uint8_t* rows, int row_count, int blocks_per_row,
                              int row_stride_bytes, const float* input, float* output);

}

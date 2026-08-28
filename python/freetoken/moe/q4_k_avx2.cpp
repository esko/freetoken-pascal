// AVX2 Q4_K arithmetic for the Issue #16 hosted backend.
// Compile this translation unit with -mavx2 -mfma; keep dispatch/scalar
// translation units on the baseline ISA.

#include "q4_k_native.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define FREETOKEN_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define FREETOKEN_AVX2_TARGET
#endif

namespace {

float half_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1Fu;
  uint32_t mantissa = bits & 0x3FFu;
  uint32_t value;
  if (exponent == 0) {
    if (mantissa == 0) {
      value = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3FFu;
      value = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    value = sign | 0x7F800000u | (mantissa << 13);
  } else {
    value = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void scale_min(const uint8_t* scales, int index, int* scale, int* minimum) {
  if (index < 4) {
    *scale = scales[index] & 0x3F;
    *minimum = scales[index + 4] & 0x3F;
    return;
  }
  *scale = (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4);
  *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

inline float q4_code(const uint8_t* block, int subblock, int lane) {
  const int group = subblock / 2;
  const int packed = block[16 + group * 32 + lane];
  return static_cast<float>((subblock & 1) ? (packed >> 4) : (packed & 0x0F));
}

}  // namespace

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) float
freetoken_q4k_dot_avx2_impl(const uint8_t* block, const float* input) {
#if defined(__x86_64__) || defined(__i386__)
  uint16_t d_bits;
  uint16_t dmin_bits;
  std::memcpy(&d_bits, block, sizeof(d_bits));
  std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
  const float d = half_to_float(d_bits);
  const float dmin = half_to_float(dmin_bits);
  __m256 accumulator = _mm256_setzero_ps();
  alignas(32) float values[8];
  for (int subblock = 0; subblock < 8; ++subblock) {
    int scale;
    int minimum;
    scale_min(block + 4, subblock, &scale, &minimum);
    const float factor = d * static_cast<float>(scale);
    const float offset = dmin * static_cast<float>(minimum);
    for (int lane = 0; lane < 32; lane += 8) {
      for (int j = 0; j < 8; ++j) {
        values[j] = factor * q4_code(block, subblock, lane + j) - offset;
      }
      const __m256 weights = _mm256_load_ps(values);
      const __m256 activations = _mm256_loadu_ps(input + subblock * 32 + lane);
      accumulator = _mm256_fmadd_ps(weights, activations, accumulator);
    }
  }
  alignas(32) float reduced[8];
  _mm256_store_ps(reduced, accumulator);
  float result = 0.0f;
  for (float value : reduced) {
    result += value;
  }
  return result;
#else
  (void)block;
  (void)input;
  return 0.0f;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) void
freetoken_q4k_decode_avx2_impl(const uint8_t* block, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  uint16_t d_bits;
  uint16_t dmin_bits;
  std::memcpy(&d_bits, block, sizeof(d_bits));
  std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
  const float d = half_to_float(d_bits);
  const float dmin = half_to_float(dmin_bits);
  alignas(32) float values[8];
  for (int subblock = 0; subblock < 8; ++subblock) {
    int scale;
    int minimum;
    scale_min(block + 4, subblock, &scale, &minimum);
    const float factor = d * static_cast<float>(scale);
    const float offset = dmin * static_cast<float>(minimum);
    for (int lane = 0; lane < 32; lane += 8) {
      for (int j = 0; j < 8; ++j) {
        values[j] = factor * q4_code(block, subblock, lane + j) - offset;
      }
      _mm256_storeu_ps(output + subblock * 32 + lane, _mm256_load_ps(values));
    }
  }
#else
  (void)block;
  (void)output;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) void
freetoken_q4k_gemv_avx2_impl(const uint8_t* rows, int row_count, int blocks_per_row,
                             int row_stride_bytes, const float* input, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  for (int row = 0; row < row_count; ++row) {
    const uint8_t* packed_row = rows + static_cast<size_t>(row) * row_stride_bytes;
    float result = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      result += freetoken_q4k_dot_avx2_impl(
          packed_row + static_cast<size_t>(block) * 144,
          input + static_cast<size_t>(block) * 256);
    }
    output[row] = result;
  }
#else
  (void)rows;
  (void)row_count;
  (void)blocks_per_row;
  (void)row_stride_bytes;
  (void)input;
  (void)output;
#endif
}

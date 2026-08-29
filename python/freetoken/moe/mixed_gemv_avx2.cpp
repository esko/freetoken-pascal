// AVX2/FMA GGML companion-format arithmetic for Issue #16.
// Compile this translation unit with -mavx2 -mfma; baseline dispatch is separate.

#include "mixed_gemv_native.h"

#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define FREETOKEN_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define FREETOKEN_AVX2_TARGET
#endif

namespace {

FREETOKEN_AVX2_TARGET float half_to_float(uint16_t bits) {
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
  union {
    uint32_t bits;
    float value;
  } result = {value};
  return result.value;
}

FREETOKEN_AVX2_TARGET uint16_t load_u16(const uint8_t* address) {
  return static_cast<uint16_t>(address[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(address[1]) << 8);
}

FREETOKEN_AVX2_TARGET uint32_t load_u32(const uint8_t* address) {
  return static_cast<uint32_t>(address[0]) |
         (static_cast<uint32_t>(address[1]) << 8) |
         (static_cast<uint32_t>(address[2]) << 16) |
         (static_cast<uint32_t>(address[3]) << 24);
}

FREETOKEN_AVX2_TARGET void scale_min(const uint8_t* scales, int index, int* scale,
                                     int* minimum) {
  if (index < 4) {
    *scale = scales[index] & 0x3F;
    *minimum = scales[index + 4] & 0x3F;
    return;
  }
  *scale = (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4);
  *minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

}  // namespace

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) float
freetoken_mixed_q5_1_dot_avx2_impl(const uint8_t* block, const float* input) {
#if defined(__x86_64__) || defined(__i386__)
  const float d = half_to_float(load_u16(block));
  const float minimum = half_to_float(load_u16(block + 2));
  const uint32_t qh = load_u32(block + 4);
  __m256 accumulator = _mm256_setzero_ps();
  const __m256 scale = _mm256_set1_ps(d);
  const __m256 offset = _mm256_set1_ps(minimum);
  const __m256i nibble_mask = _mm256_set1_epi32(0x0F);
  const __m256i bit_mask = _mm256_set1_epi32(1);
  const __m256i bit_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (int lane = 0; lane < 32; lane += 8) {
    // Q5_1 stores eight adjacent low nibbles in one byte load and the fifth
    // bit in the corresponding eight bits of qh.  Expand both byte vectors in
    // AVX2 instead of constructing eight scalar float values per chunk.
    const __m128i packed = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(block + 8 + (lane & 15)));
    __m256i codes = _mm256_cvtepu8_epi32(packed);
    if (lane < 16) {
      codes = _mm256_and_si256(codes, nibble_mask);
    } else {
      codes = _mm256_and_si256(_mm256_srli_epi32(codes, 4), nibble_mask);
    }
    const __m256i shifted_bits = _mm256_srlv_epi32(
        _mm256_set1_epi32(static_cast<int>(qh >> lane)), bit_shifts);
    const __m256i high_codes = _mm256_slli_epi32(
        _mm256_and_si256(shifted_bits, bit_mask), 4);
    codes = _mm256_or_si256(codes, high_codes);
    const __m256 values = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), scale, offset);
    accumulator = _mm256_fmadd_ps(
        values, _mm256_loadu_ps(input + lane), accumulator);
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
freetoken_mixed_q5_1_decode_avx2_impl(const uint8_t* block, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  const float d = half_to_float(load_u16(block));
  const float minimum = half_to_float(load_u16(block + 2));
  const uint32_t qh = load_u32(block + 4);
  alignas(32) float values[8];
  for (int lane = 0; lane < 32; lane += 8) {
    for (int j = 0; j < 8; ++j) {
      const int index = lane + j;
      const uint8_t packed = block[8 + (index & 15)];
      const int low_code = index < 16 ? (packed & 0x0F) : (packed >> 4);
      const int code = low_code | static_cast<int>(((qh >> index) & 1u) << 4);
      values[j] = static_cast<float>(code) * d + minimum;
    }
    _mm256_storeu_ps(output + lane, _mm256_load_ps(values));
  }
#else
  (void)block;
  (void)output;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) float
freetoken_mixed_q8_0_dot_avx2_impl(const uint8_t* block, const float* input) {
#if defined(__x86_64__) || defined(__i386__)
  const float d = half_to_float(load_u16(block));
  __m256 accumulator = _mm256_setzero_ps();
  alignas(32) float values[8];
  for (int lane = 0; lane < 32; lane += 8) {
    for (int j = 0; j < 8; ++j) {
      values[j] = static_cast<float>(static_cast<int8_t>(block[2 + lane + j])) * d;
    }
    accumulator = _mm256_fmadd_ps(
        _mm256_load_ps(values), _mm256_loadu_ps(input + lane), accumulator);
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
freetoken_mixed_q8_0_decode_avx2_impl(const uint8_t* block, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  const float d = half_to_float(load_u16(block));
  alignas(32) float values[8];
  for (int lane = 0; lane < 32; lane += 8) {
    for (int j = 0; j < 8; ++j) {
      values[j] = static_cast<float>(static_cast<int8_t>(block[2 + lane + j])) * d;
    }
    _mm256_storeu_ps(output + lane, _mm256_load_ps(values));
  }
#else
  (void)block;
  (void)output;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) float
freetoken_mixed_q5_k_dot_avx2_impl(const uint8_t* block, const float* input) {
#if defined(__x86_64__) || defined(__i386__)
  const float d = half_to_float(load_u16(block));
  const float dmin = half_to_float(load_u16(block + 2));
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
        const int index = lane + j;
        const uint8_t packed = block[48 + (subblock / 2) * 32 + index];
        const int low_code = (subblock & 1) != 0 ? (packed >> 4) : (packed & 0x0F);
        const int high_code = ((block[16 + index] >> subblock) & 1) << 4;
        values[j] = static_cast<float>(low_code | high_code) * factor - offset;
      }
      accumulator = _mm256_fmadd_ps(
          _mm256_load_ps(values), _mm256_loadu_ps(input + subblock * 32 + lane), accumulator);
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
freetoken_mixed_q5_k_decode_avx2_impl(const uint8_t* block, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  const float d = half_to_float(load_u16(block));
  const float dmin = half_to_float(load_u16(block + 2));
  alignas(32) float values[8];
  for (int subblock = 0; subblock < 8; ++subblock) {
    int scale;
    int minimum;
    scale_min(block + 4, subblock, &scale, &minimum);
    const float factor = d * static_cast<float>(scale);
    const float offset = dmin * static_cast<float>(minimum);
    for (int lane = 0; lane < 32; lane += 8) {
      for (int j = 0; j < 8; ++j) {
        const int index = lane + j;
        const uint8_t packed = block[48 + (subblock / 2) * 32 + index];
        const int low_code = (subblock & 1) != 0 ? (packed >> 4) : (packed & 0x0F);
        const int high_code = ((block[16 + index] >> subblock) & 1) << 4;
        values[j] = static_cast<float>(low_code | high_code) * factor - offset;
      }
      _mm256_storeu_ps(output + subblock * 32 + lane, _mm256_load_ps(values));
    }
  }
#else
  (void)block;
  (void)output;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) int
freetoken_mixed_q5_1_gemv_avx2_impl(const uint8_t* rows, int row_count, int blocks_per_row,
                                    int row_stride_bytes, const float* input, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  if (rows == nullptr || input == nullptr || output == nullptr || row_count <= 0 ||
      blocks_per_row <= 0 || blocks_per_row > 0x7FFFFFFF / 24 ||
      row_stride_bytes != blocks_per_row * 24) {
    return -1;
  }
  const __m256i nibble_mask = _mm256_set1_epi32(0x0F);
  const __m256i bit_mask = _mm256_set1_epi32(1);
  const __m256i bit_shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  for (int row = 0; row < row_count; ++row) {
    const uint8_t* packed_row = rows + static_cast<size_t>(row) * row_stride_bytes;
    __m256 accumulator = _mm256_setzero_ps();
    for (int block = 0; block < blocks_per_row; ++block) {
      const uint8_t* packed_block = packed_row + static_cast<size_t>(block) * 24;
      const float* block_input = input + static_cast<size_t>(block) * 32;
      const float d = half_to_float(load_u16(packed_block));
      const float minimum = half_to_float(load_u16(packed_block + 2));
      const uint32_t qh = load_u32(packed_block + 4);
      const __m256 scale = _mm256_set1_ps(d);
      const __m256 offset = _mm256_set1_ps(minimum);
      for (int lane = 0; lane < 32; lane += 8) {
        const __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
            packed_block + 8 + (lane & 15)));
        __m256i codes = _mm256_cvtepu8_epi32(packed);
        if (lane < 16) {
          codes = _mm256_and_si256(codes, nibble_mask);
        } else {
          codes = _mm256_and_si256(_mm256_srli_epi32(codes, 4), nibble_mask);
        }
        const __m256i shifted_bits = _mm256_srlv_epi32(
            _mm256_set1_epi32(static_cast<int>(qh >> lane)), bit_shifts);
        const __m256i high_codes = _mm256_slli_epi32(
            _mm256_and_si256(shifted_bits, bit_mask), 4);
        codes = _mm256_or_si256(codes, high_codes);
        const __m256 values = _mm256_fmadd_ps(_mm256_cvtepi32_ps(codes), scale, offset);
        accumulator = _mm256_fmadd_ps(
            values, _mm256_loadu_ps(block_input + lane), accumulator);
      }
    }
    alignas(32) float reduced[8];
    _mm256_store_ps(reduced, accumulator);
    float result = 0.0f;
    for (float value : reduced) {
      result += value;
    }
    output[row] = result;
  }
  return 0;
#else
  (void)rows;
  (void)row_count;
  (void)blocks_per_row;
  (void)row_stride_bytes;
  (void)input;
  (void)output;
  return -1;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) int
freetoken_mixed_q8_0_gemv_avx2_impl(const uint8_t* rows, int row_count, int blocks_per_row,
                                    int row_stride_bytes, const float* input, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  if (rows == nullptr || input == nullptr || output == nullptr || row_count <= 0 ||
      blocks_per_row <= 0 || blocks_per_row > 0x7FFFFFFF / 34 ||
      row_stride_bytes != blocks_per_row * 34) {
    return -1;
  }
  for (int row = 0; row < row_count; ++row) {
    const uint8_t* packed_row = rows + static_cast<size_t>(row) * row_stride_bytes;
    float result = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      result += freetoken_mixed_q8_0_dot_avx2_impl(
          packed_row + static_cast<size_t>(block) * 34,
          input + static_cast<size_t>(block) * 32);
    }
    output[row] = result;
  }
  return 0;
#else
  (void)rows;
  (void)row_count;
  (void)blocks_per_row;
  (void)row_stride_bytes;
  (void)input;
  (void)output;
  return -1;
#endif
}

extern "C" FREETOKEN_AVX2_TARGET __attribute__((visibility("default"))) int
freetoken_mixed_q5_k_gemv_avx2_impl(const uint8_t* rows, int row_count, int blocks_per_row,
                                    int row_stride_bytes, const float* input, float* output) {
#if defined(__x86_64__) || defined(__i386__)
  if (rows == nullptr || input == nullptr || output == nullptr || row_count <= 0 ||
      blocks_per_row <= 0 || blocks_per_row > 0x7FFFFFFF / 176 ||
      row_stride_bytes != blocks_per_row * 176) {
    return -1;
  }
  for (int row = 0; row < row_count; ++row) {
    const uint8_t* packed_row = rows + static_cast<size_t>(row) * row_stride_bytes;
    float result = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
      result += freetoken_mixed_q5_k_dot_avx2_impl(
          packed_row + static_cast<size_t>(block) * 176,
          input + static_cast<size_t>(block) * 256);
    }
    output[row] = result;
  }
  return 0;
#else
  (void)rows;
  (void)row_count;
  (void)blocks_per_row;
  (void)row_stride_bytes;
  (void)input;
  (void)output;
  return -1;
#endif
}

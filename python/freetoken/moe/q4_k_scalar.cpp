// Baseline scalar Q4_K arithmetic for the Issue #16 hosted backend.
// This translation unit must be compiled without any -mavx* option.

#include "q4_k_native.h"

#include <cstdint>
#include <cstring>

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

extern "C" __attribute__((visibility("default"))) float
freetoken_q4k_dot_scalar(const uint8_t* block, const float* input) {
  uint16_t d_bits;
  uint16_t dmin_bits;
  std::memcpy(&d_bits, block, sizeof(d_bits));
  std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
  const float d = half_to_float(d_bits);
  const float dmin = half_to_float(dmin_bits);
  float accumulator = 0.0f;
  for (int subblock = 0; subblock < 8; ++subblock) {
    int scale;
    int minimum;
    scale_min(block + 4, subblock, &scale, &minimum);
    const float factor = d * static_cast<float>(scale);
    const float offset = dmin * static_cast<float>(minimum);
    for (int lane = 0; lane < 32; ++lane) {
      accumulator += (factor * q4_code(block, subblock, lane) - offset) *
                     input[subblock * 32 + lane];
    }
  }
  return accumulator;
}

extern "C" __attribute__((visibility("default"))) void
freetoken_q4k_decode_scalar(const uint8_t* block, float* output) {
  uint16_t d_bits;
  uint16_t dmin_bits;
  std::memcpy(&d_bits, block, sizeof(d_bits));
  std::memcpy(&dmin_bits, block + 2, sizeof(dmin_bits));
  const float d = half_to_float(d_bits);
  const float dmin = half_to_float(dmin_bits);
  for (int subblock = 0; subblock < 8; ++subblock) {
    int scale;
    int minimum;
    scale_min(block + 4, subblock, &scale, &minimum);
    const float factor = d * static_cast<float>(scale);
    const float offset = dmin * static_cast<float>(minimum);
    for (int lane = 0; lane < 32; ++lane) {
      output[subblock * 32 + lane] = factor * q4_code(block, subblock, lane) - offset;
    }
  }
}

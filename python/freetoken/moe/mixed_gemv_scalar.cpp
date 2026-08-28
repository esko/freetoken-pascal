// Scalar GGML companion-format arithmetic for Issue #16.
// The layout follows llama.cpp commit eaf93765572e794b8e3754fe45adbe12d381e997.
// This translation unit is compiled without AVX flags and is the native oracle.

#include "mixed_gemv_native.h"

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

uint16_t load_u16(const uint8_t* address) {
  uint16_t value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

uint32_t load_u32(const uint8_t* address) {
  uint32_t value;
  std::memcpy(&value, address, sizeof(value));
  return value;
}

float q5_1_value(const uint8_t* block, int index) {
  const float d = half_to_float(load_u16(block));
  const float minimum = half_to_float(load_u16(block + 2));
  const uint32_t qh = load_u32(block + 4);
  const int lane = index & 15;
  const int high_bit = index < 16 ? lane : lane + 16;
  const uint8_t packed = block[8 + lane];
  const int low_code = index < 16 ? (packed & 0x0F) : (packed >> 4);
  const int code = low_code | static_cast<int>(((qh >> high_bit) & 1u) << 4);
  return static_cast<float>(code) * d + minimum;
}

float q8_0_value(const uint8_t* block, int index) {
  const float d = half_to_float(load_u16(block));
  const int8_t code = static_cast<int8_t>(block[2 + index]);
  return static_cast<float>(code) * d;
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

float q5_k_value(const uint8_t* block, int index) {
  const float d = half_to_float(load_u16(block));
  const float dmin = half_to_float(load_u16(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* ql = block + 48;
  const int subblock = index / 32;
  const int lane = index & 31;
  int scale;
  int minimum;
  scale_min(scales, subblock, &scale, &minimum);
  const uint8_t packed = ql[(subblock / 2) * 32 + lane];
  const int low_code = (subblock & 1) != 0 ? (packed >> 4) : (packed & 0x0F);
  const int high_code = ((qh[lane] >> subblock) & 1) << 4;
  return (static_cast<float>(low_code | high_code) * d * static_cast<float>(scale)) -
         (dmin * static_cast<float>(minimum));
}

}  // namespace

extern "C" __attribute__((visibility("default"))) float
freetoken_mixed_q5_1_dot_scalar(const uint8_t* block, const float* input) {
  float result = 0.0f;
  for (int index = 0; index < 32; ++index) {
    result += q5_1_value(block, index) * input[index];
  }
  return result;
}

extern "C" __attribute__((visibility("default"))) void
freetoken_mixed_q5_1_decode_scalar(const uint8_t* block, float* output) {
  for (int index = 0; index < 32; ++index) {
    output[index] = q5_1_value(block, index);
  }
}

extern "C" __attribute__((visibility("default"))) float
freetoken_mixed_q8_0_dot_scalar(const uint8_t* block, const float* input) {
  float result = 0.0f;
  for (int index = 0; index < 32; ++index) {
    result += q8_0_value(block, index) * input[index];
  }
  return result;
}

extern "C" __attribute__((visibility("default"))) void
freetoken_mixed_q8_0_decode_scalar(const uint8_t* block, float* output) {
  for (int index = 0; index < 32; ++index) {
    output[index] = q8_0_value(block, index);
  }
}

extern "C" __attribute__((visibility("default"))) float
freetoken_mixed_q5_k_dot_scalar(const uint8_t* block, const float* input) {
  float result = 0.0f;
  for (int index = 0; index < 256; ++index) {
    result += q5_k_value(block, index) * input[index];
  }
  return result;
}

extern "C" __attribute__((visibility("default"))) void
freetoken_mixed_q5_k_decode_scalar(const uint8_t* block, float* output) {
  for (int index = 0; index < 256; ++index) {
    output[index] = q5_k_value(block, index);
  }
}

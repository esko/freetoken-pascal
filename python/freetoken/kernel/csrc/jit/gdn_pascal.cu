// SPDX-License-Identifier: MIT
// FreeToken-Pascal standalone FP32 GatedDeltaNet recurrence for sm_61.
//
// The recurrence is adapted from the algorithm in
// poisonxa16/pxq_llama.cpp@d34d74e93b95761e67a17a649cf2faf039e7888e
// (MIT; see manifests/upstreams.yaml and NOTICE).  This file intentionally
// does not copy the donor's fusion or convolution code.  FreeToken
// owns the ragged slot and state-pool contract here.
//
// State contract: state_pool is contiguous [slots, V_heads, K, V], where each
// matrix is addressed as [key, value].  beta is already pre-sigmoided
// (sigma(beta_logits))
// and is consumed as supplied; this kernel never applies the gate.  g is the
// log-decay produced by the caller and is exponentiated once per step.

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace {

template <int kHeadDim> struct PascalGdnParams {
  const float *__restrict__ q;
  const float *__restrict__ k;
  const float *__restrict__ v;
  const float *__restrict__ g;
  const float *__restrict__ beta;
  float *__restrict__ state_pool;
  const std::int32_t *__restrict__ slot_indices;
  const std::int32_t *__restrict__ cu_seqlens;
  float *__restrict__ output;
  std::int64_t num_k_heads;
  std::int64_t num_v_heads;
  std::int64_t state_slots;
  std::int64_t num_requests;
};

template <int kHeadDim>
__global__ void pascal_gdn_recurrence_f32(
    const PascalGdnParams<kHeadDim> params) {
  static_assert(kHeadDim == 64 || kHeadDim == 128,
                "Pascal GDN only supports D64 and D128");

  const std::int64_t request =
      static_cast<std::int64_t>(blockIdx.x) / params.num_v_heads;
  const std::int64_t value_head =
      static_cast<std::int64_t>(blockIdx.x) % params.num_v_heads;
  if (request >= params.num_requests) {
    return;
  }

  // GQA repeats each key head contiguously across value heads, matching
  // repeat_interleave in the independent FreeToken reference recurrence.
  const std::int64_t gqa_ratio = params.num_v_heads / params.num_k_heads;
  const std::int64_t key_head = value_head / gqa_ratio;
  const std::int64_t start = params.cu_seqlens[request];
  const std::int64_t end = params.cu_seqlens[request + 1];
  const std::int64_t slot = params.slot_indices[request];

  // The adapter validates slot bounds and duplicate ownership before launch.
  // Keeping the address arithmetic explicit prevents accidental flattening of
  // [K,V] into the donor's [V,K] state convention.
  float *const state = params.state_pool +
                       ((slot * params.num_v_heads + value_head) * kHeadDim *
                        kHeadDim);

  __shared__ float q_shared[kHeadDim];
  __shared__ float k_shared[kHeadDim];

  const int thread = static_cast<int>(threadIdx.x);
  for (std::int64_t token = start; token < end; ++token) {
    if (thread < kHeadDim) {
      const auto q_offset =
          (token * params.num_k_heads + key_head) * kHeadDim + thread;
      q_shared[thread] = params.q[q_offset];
      k_shared[thread] = params.k[q_offset];
    }
    __syncthreads();

    // Normalize in FP32 and apply the key-head scale to q, as the reference
    // path does.  A single thread performs this tiny D64/D128 operation so no
    // extra reduction kernel or donor-specific launch assumption is needed on Pascal.
    if (thread == 0) {
      float q_norm = 0.0f;
      float k_norm = 0.0f;
      for (int key = 0; key < kHeadDim; ++key) {
        q_norm += q_shared[key] * q_shared[key];
        k_norm += k_shared[key] * k_shared[key];
      }
      const float q_scale = rsqrtf(q_norm + 1.0e-6f) / sqrtf(kHeadDim);
      const float k_scale = rsqrtf(k_norm + 1.0e-6f);
      for (int key = 0; key < kHeadDim; ++key) {
        q_shared[key] *= q_scale;
        k_shared[key] *= k_scale;
      }
    }
    __syncthreads();

    if (thread < kHeadDim) {
      const std::int64_t value_offset =
          (token * params.num_v_heads + value_head) * kHeadDim + thread;
      const float beta_value =
          params.beta[token * params.num_v_heads + value_head];
      const float decay = expf(params.g[token * params.num_v_heads + value_head]);

      // One thread owns one V column.  This makes every [K,V] state element
      // have one writer while preserving the sequential recurrence per slot.
      float memory = 0.0f;
      for (int key = 0; key < kHeadDim; ++key) {
        memory += decay * state[key * kHeadDim + thread] * k_shared[key];
      }
      const float delta = (params.v[value_offset] - memory) * beta_value;

      float result = 0.0f;
      for (int key = 0; key < kHeadDim; ++key) {
        const auto state_index = key * kHeadDim + thread;
        const float next = decay * state[state_index] + k_shared[key] * delta;
        state[state_index] = next;
        result += next * q_shared[key];
      }
      params.output[value_offset] = result;
    }
    __syncthreads();
  }
}

template <int kHeadDim> struct PascalGdnKernel {
  static void run(const tvm::ffi::TensorView q,
                  const tvm::ffi::TensorView k,
                  const tvm::ffi::TensorView v,
                  const tvm::ffi::TensorView g,
                  const tvm::ffi::TensorView beta,
                  const tvm::ffi::TensorView state_pool,
                  const tvm::ffi::TensorView slot_indices,
                  const tvm::ffi::TensorView cu_seqlens,
                  const tvm::ffi::TensorView output) {
    using namespace host;

    auto T = SymbolicSize{"T"};
    auto HK = SymbolicSize{"HK"};
    auto HV = SymbolicSize{"HV"};
    auto S = SymbolicSize{"S"};
    auto B = SymbolicSize{"B"};
    auto device = SymbolicDevice{};

    TensorMatcher({T, HK, kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(q);
    TensorMatcher({T, HK, kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(k);
    TensorMatcher({T, HV, kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(v);
    TensorMatcher({T, HV})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(g);
    TensorMatcher({T, HV})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(beta);
    TensorMatcher({S, HV, kHeadDim, kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(state_pool);
    TensorMatcher({B})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(slot_indices);
    TensorMatcher({-1})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(cu_seqlens);
    TensorMatcher({T, HV, kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(output);

    RuntimeCheck(cu_seqlens.size(0) == B.unwrap() + 1,
                 "Pascal GDN cu_seqlens must have B + 1 entries");
    RuntimeCheck(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                     g.is_contiguous() && beta.is_contiguous() &&
                     state_pool.is_contiguous() && slot_indices.is_contiguous() &&
                     cu_seqlens.is_contiguous() && output.is_contiguous(),
                 "Pascal GDN tensors must be contiguous");
    RuntimeCheck(HK.unwrap() > 0 && HV.unwrap() > 0 &&
                     HV.unwrap() % HK.unwrap() == 0,
                 "Pascal GDN requires positive divisible GQA head counts");
    RuntimeCheck(B.unwrap() > 0, "Pascal GDN requires at least one request");
    RuntimeCheck(T.unwrap() > 0, "Pascal GDN requires at least one token");
    RuntimeCheck(B.unwrap() <= std::numeric_limits<std::int32_t>::max(),
                 "Pascal GDN request count exceeds int32 range");

    const auto num_blocks = static_cast<std::uint64_t>(B.unwrap()) *
                            static_cast<std::uint64_t>(HV.unwrap());
    RuntimeCheck(num_blocks <= std::numeric_limits<unsigned int>::max(),
                 "Pascal GDN launch grid exceeds CUDA limits");
    const auto params = PascalGdnParams<kHeadDim>{
        .q = static_cast<const float *>(q.data_ptr()),
        .k = static_cast<const float *>(k.data_ptr()),
        .v = static_cast<const float *>(v.data_ptr()),
        .g = static_cast<const float *>(g.data_ptr()),
        .beta = static_cast<const float *>(beta.data_ptr()),
        .state_pool = static_cast<float *>(state_pool.data_ptr()),
        .slot_indices = static_cast<const std::int32_t *>(slot_indices.data_ptr()),
        .cu_seqlens = static_cast<const std::int32_t *>(cu_seqlens.data_ptr()),
        .output = static_cast<float *>(output.data_ptr()),
        .num_k_heads = HK.unwrap(),
        .num_v_heads = HV.unwrap(),
        .state_slots = S.unwrap(),
        .num_requests = B.unwrap(),
    };

    LaunchKernel(static_cast<unsigned int>(num_blocks), 128, device.unwrap())(
        pascal_gdn_recurrence_f32<kHeadDim>, params);
  }
};

// Explicit instantiation is intentional: the standalone sm_61 source census
// must contain both device variants before a runtime JIT wrapper specializes a
// launch configuration.
template struct PascalGdnKernel<64>;
template struct PascalGdnKernel<128>;

} // namespace

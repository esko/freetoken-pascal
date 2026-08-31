# Pascal compatibility profile

FreeToken-Pascal supports CUDA compute capability 6.0 or newer and targets Tesla P4 (`sm_61`)
with the pinned CUDA 12.6 environment. CUDA 13 is rejected on pre-Turing devices because its
toolchain no longer supports Pascal. Devices older than compute capability 6.0 fail at startup.

The machine-readable inventory is `manifests/gpu-compatibility.json`. At startup the engine logs
the device, compute capability, CUDA runtime, selected kernel modes, and whether each optional
kernel package is enabled. A package being importable is not sufficient: its minimum architecture
must also be satisfied before a backend can select it.

## Pascal fallbacks

| Feature | Modern path | `sm_61` path |
| --- | --- | --- |
| JIT kernel parameters | `__grid_constant__` | ordinary by-value parameter |
| Global load hint | `ld.global.L1::no_allocate` | ordinary `ld.global` |
| GELU tanh | Triton approximate tanh | libdevice tanh |
| MoE token alignment | fused atomic kernel | staged atomic-free kernel |
| top-k/top-p sampling | Triton atomic threshold search | Torch sort and inverse CDF |
| attention tiles | large shared-memory tiles | device-budgeted descending tile ladder |
| QSA gathered-row dot products | Triton BF16/FP16 | exact Torch FP32 reference |

FlashInfer and `sgl_kernel` are disabled below compute capability 7.5. OpenAI
`triton_kernels` is disabled below 7.0. Their in-tree fallbacks are selected before launch and
reported; explicit requests for an incompatible backend fail with an actionable error.

## Validation

Run H0 policy, inventory, and failure-path tests with:

```bash
PYTHONPATH=python pytest -q tests/project/test_pascal_compatibility.py
```

Run the GPU-free H1 compile and selector gate in the pinned CUDA 12.6 image with:

```bash
scripts/ci/verify_cuda126.sh
```

H1 compiles every shipping CUDA translation unit for `sm_61`, inspects the resulting device code,
and runs the attention tile selectors without accessing a GPU. H1 does not establish runtime
correctness. Activation, sampling, MoE, attention, and pinned-memory parity must still run on a real
Tesla P4 under H2. Issue #9 has established bounded per-card allocation and arithmetic on the
installed cards; sustained thermal qualification and runner provisioning remain open.

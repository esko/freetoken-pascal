# Hardware target and deferred validation

## Known host

| Component | Target |
|---|---|
| Server | Fujitsu PRIMERGY RX2530 M1 |
| CPU | 2 × Xeon E5-2673 v3 |
| ISA | AVX2, no AVX-512/AMX |
| NUMA | two CPU sockets |
| Memory | about 128 GB ECC DDR4 |
| Storage | 1 TB PCIe M.2 NVMe for models |
| GPU | 2 × Tesla P4, 8 GB, `sm_61` |
| Interconnect | PCIe, no NVLink |
| OS | Ubuntu Server 26.04 LTS |

The P4 cards are not yet available. No issue requiring H2 or H3 evidence may be closed until the cards are installed and the self-hosted runner records the actual topology.

## Immediate work before GPU arrival

The following can and should be completed now:

- repository import and upstream pinning;
- CUDA 12.6 container and `sm_61` compilation;
- host simulators and CPU tests;
- Qwen4 and GGUF parsers/loaders;
- tensor-census tools;
- PLE mmap and page-cache tests;
- AVX2 expert kernels and parity tests;
- cache policy simulation from captured or synthetic traces;
- API/config/metrics contracts;
- CI and benchmark harnesses;
- tiny-model tests on available non-Pascal hardware where architecture-independent.

## Arrival checklist

When the P4s arrive:

1. Update firmware and install the cards with correct airflow.
2. Record:
   - `nvidia-smi -q`
   - `nvidia-smi topo -m`
   - `lspci -tv`
   - `numactl -H`
   - PCIe link width/speed
   - GPU power and application-clock capabilities
   - thermal behavior under a sustained load
3. Verify both GPUs report compute capability 6.1 and 8 GB VRAM.
4. Map each GPU to its local CPU/NUMA node.
5. Install the pinned driver and CUDA 12.6 toolchain.
6. Register the self-hosted GitHub runner with labels:
   `self-hosted`, `linux`, `x64`, `cuda`, `sm61`.
7. Run only the H2 smoke gate first.
8. Add the second card only after single-card correctness passes.
9. Attach inventory artifacts to the hardware qualification issue.

## Memory rules

- Do not add RAM solely for model capacity before measuring.
- Never permit swap during a performance run.
- Keep ZFS ARC bounded during inference qualification.
- PLE may be backed by NVMe, but warm and cold behavior must be measured separately.
- Excessive pinned memory can destabilize the host; use bounded staging or a measured pinned bank.
- Memory pages for a GPU-owned layer should prefer the GPU-local NUMA node unless A/B evidence favors interleaving.

## Thermal rules

Tesla P4 is passively cooled. A hardware result is invalid if clocks throttle because of airflow or power limits. Record temperature, clocks and power during every release benchmark.

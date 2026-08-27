# Product scope

## Product statement

FreeToken-Pascal v1 is a downstream FreeToken runtime optimized for **text-only Qwen3.8-Flash-Next inference on a dual NVIDIA Tesla P4 server**. It is designed for a large host-memory, small-VRAM system where the complete low-bit expert bank remains available in RAM while the GPUs accelerate the continuously active model path and a dynamically selected hot subset of routed experts.

The product is not a generic inference platform. Its v1 purpose is to maximize reliable throughput on the owner's actual server without changing the model topology or sacrificing experts.

## Reference hardware

- Fujitsu PRIMERGY RX2530 M1
- 2 × Intel Xeon E5-2673 v3, 24 physical cores / 48 threads total
- approximately 128 GB ECC DDR4, 8 memory channels across two NUMA nodes
- 2 × Tesla P4, 8 GB each, Pascal `sm_61` — **not installed yet**
- 1 TB PCIe M.2 NVMe for model and PLE storage
- Ubuntu Server 26.04 LTS
- CUDA 12.6-class toolchain required for Pascal
- no NVLink; PCIe/NUMA topology must be measured after installation

## In-scope v1 capabilities

### Model

- Qwen3.8-Flash-Next text backbone
- Gated DeltaNet, QSA, hyperconnections, PLE and routed/shared experts
- native tokenizer and chat template
- deterministic state reset and restore
- 32K required validation context
- 128K release validation context
- 262K native-context qualification when memory and current upstream correctness permit; failure to qualify must be documented, not hidden

### Model storage and quantization

- GGUF/K/I weight loading
- Unsloth `UD-Q4_K_XL` as the primary reference artifact
- at least one smaller 3-bit artifact for comparison
- heterogeneous quant types across ordinary tensors and expert layers
- expert gate/up and down banks may use different quant types
- PLE mmap-backed from NVMe with explicit warm/cold controls
- exact tensor census and model checksums recorded for every benchmark

### Execution

- CPU-only or cache-disabled reference mode
- AVX2 low-bit CPU routed-expert execution
- one-P4 bring-up mode
- dual-P4 release mode
- deterministic layer ownership per GPU
- fixed-address per-GPU expert cache
- LFRU or measured successor policy
- asynchronous cache fill
- concurrent CPU and GPU execution for one MoE operation
- contention-aware split of current misses
- separate decode and prefill policies
- NUMA-aware host placement and worker pools
- automatic fallback when hybrid execution is slower or unsafe

### Serving and observability

- OpenAI-compatible text API
- streaming, cancellation and bounded request resources
- one primary interactive request; limited concurrency may be supported but is not the primary optimization target
- metrics for cache hits, misses, evictions, CPU/GPU expert counts, transfer bytes/times, scheduler decisions, NUMA placement, PP/TG and memory
- reproducible benchmark commands and machine-readable results
- Docker/Compose deployment compatible with the server's standard stack
- crash-safe logs and actionable unsupported-configuration errors

## Out of scope for v1

- vision tower and image serving
- Windows support
- GLM-5.3 or other model families
- MTP, DFlash, n-gram speculative decoding and speculative prefill
- distributed multi-node serving
- expert parallelism across more than two local P4s
- training, fine-tuning, REAP pruning or topology changes
- a custom quantization research program beyond formats required for the target artifacts
- public SaaS multi-tenancy
- replacing upstream FreeToken's general architecture

These may be added only after v1 release or by an accepted ADR that changes scope and backlog.

## Release success

The project is successful when the release criteria in `release-criteria.md` pass on the target server and the full system can be rebuilt from a clean checkout using pinned sources and documented commands.

# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |

### MoE offload

See [models.md](models.md#moe-backends) for what each backend does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-backend` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family backends) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--ple-artifact-path` | unset | Explicit dedicated PLE artifact directory; omission preserves the source-model loader and invalid artifacts never fall back silently |
| `--ple-warm-mode` | `cold` | PLE policy: `cold`, `page-cache-warm`, selected-row `targeted`, transitional GGUF `full-model-warm`, or dedicated-artifact `full-ple-warm` |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the compiled cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, or a fraction) |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

The standalone Qwen GGUF CPU bridge is not registered by `ft serve` yet, so
`--moe-cpu-threads` does not change that bridge.  Its explicit Python
`num_threads` argument uses a separate safe policy: omitted or `0` means one
serial worker, while a positive value must fit within the process's
affinity-visible physical-core capacity.  The bridge reports both the selected
policy and the actual participating thread partitions after decode.  For positive
requests its Q4 runner pins only its internally owned workers to distinct planned
CPUs and verifies the singleton read-back lazily at first participating decode.
Pin/read-back failure drains the request and reruns the serial reference path;
telemetry reports the requested and observed CPUs, errors, and explicit fallback.
Caller-supplied pools are not accepted with this explicit plan.  This is an H0
CPU-affinity check only; it does not change the owner mask or claim NUMA placement.

For the compiled CPU/hybrid executor, `0` plans one worker per visible physical
core using the process affinity mask and a positive count is exact; requests above
the visible physical-core capacity fail before the native pool is built. Flag-sync
may reserve one additional visible core for its coordinator and falls back to the
host-function path when no spare core exists. Startup telemetry distinguishes the
planned CPU IDs from native `verified` or `fallback` results; it never reports
successful affinity without an exact read-back. With an explicit worker count,
that reservation is intentional: the coordinator consumes one additional core
and the requested worker count remains exact. The `flag_sync_requested` and
`flag_sync` telemetry fields distinguish the requested optimization from the
native-applied mode; a missing, failed, or timed-out coordinator is stopped and
the host-function path is used. If worker affinity startup itself times out,
the worker pool is terminally unusable and serving construction fails rather
than submitting work to an unready pool. The native five-second wait bounds
startup reporting only; teardown joins native threads and the H1 process timeout
is the outer protection. Teardown never marks incomplete flag work complete.

### Host expert-bank policy (Issue #18 H0/H1 slice)

These flags are opt-in.
When `--host-bank-strategy` is omitted, `EngineConfig.host_bank_policy` is `None` and the legacy expert-bank loader behavior is preserved exactly.
The explicit policy is preflighted from FTW metadata before host-bank allocation or shard reads.

| Flag | Default | Meaning |
|---|---|---|
| `--host-bank-strategy` | omitted (`None`) | `pinned` is the only operational Engine strategy in this slice; `pageable` and `bounded-staging` are preflight-only and fail clearly if requested for serving |
| `--host-bank-max-pinned-bytes` | unset | Required finite byte limit for `pinned`; the page-rounded complete FTW bank set must fit before loading |
| `--host-bank-max-staging-bytes` | unset | Required finite bound for `bounded-staging` preflight |
| `--host-bank-staging-bytes`, `--host-bank-staging-slots` | 0, 2 | Fixed staging-ring geometry for the preflight primitive; the serving transfer path is not wired in this slice |
| `--host-bank-selected-layers` | all | Metadata accepted by the policy foundation, but Engine serving rejects selective pinned residency until per-layer routing is wired |
| `--host-bank-numa-policy`, `--host-bank-numa-node` | `preferred`, unset | Record NUMA intent; placement remains disabled unless `--host-bank-enforce-numa-placement` is set |
| `--host-bank-enforce-numa-placement` | off | Opt in to Linux x86_64 `mbind` for policy-owned FTW anonymous mmap banks; `bind` fails closed, while preferred/interleave report fallback on unavailable placement |
| `--host-bank-require-no-swap` | off | Require a clear read-only procfs swap probe before policy-owned FTW preparation; active/process swap or unavailable/ambiguous data fails closed |

Pinned policy never honors `FREETOKEN_SKIP_BANK_PIN=1`; serving fails closed rather than reporting requested pinning as applied.
Unsupported dummy, custom-provider, and non-FTW paths reject an explicit policy instead of silently falling back.
`--host-bank-require-no-swap` is an explicit opt-in for any host-bank strategy. It reports `VmSwap`, `SwapTotal`, `SwapFree`, active swap devices, probe source/errors, raw `swap_status`, and `no_swap_observed` in host-bank accounting. The check is read-only and point-in-time; it does not disable swap or guarantee that later model execution cannot swap. Without this flag, policy preparation does not probe procfs and reports `swap_status=not-requested`.
NUMA placement is likewise opt-in and only applies to policy-owned FTW mappings before their first touch; it does not bind threads, change system policy, or make a full-model residency claim. Enforced `bind` requires an allowed `--host-bank-numa-node`; enforced `preferred` with no node reports an explicit fallback without issuing `mbind`, while `interleave` without one targets all online allowed nodes. With `--host-bank-numa-sample-residency`, the bounded self-only `move_pages` sample reports observed node counts and `verified`, `partial`, or `unavailable` status.

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--shard-gib 8] [--gpu <uuid-or-index>]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. `--moe-backend
offload` (default) packs experts into offload banks; `--moe-backend triton`
keeps them dense for resident serving. See the FTW caveats in
[models.md](models.md#notes).

## ft bench bw

```bash
ft bench bw                       # once per GPU
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
ft bench bw --gpu 1               # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes a
profile that `ft serve --moe-backend auto` and `--moe-hybrid-max-fetch -1` then read.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`.
- Keyed on expert format + GPU, so a profile from other hardware is ignored rather than
  misapplied. An older single `benchbw.json` still counts if its GPU name matches.
- What to measure: `--dtype`, `--model`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets the call: recommend hybrid when CPU bandwidth beats PCIe
  by that factor.

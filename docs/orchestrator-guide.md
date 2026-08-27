# Main-agent orchestrator guide

## Role

The main agent owns delivery of the complete v1 scope. It coordinates specialist agents, issues, branches, reviews, hardware gates and evidence. It must not merely delegate and summarize; it is responsible for integration correctness and for completing every release gate.

## Required reading order

1. `AGENTS.md`
2. `docs/product-scope.md`
3. `docs/architecture.md`
4. `docs/implementation-plan.md`
5. `docs/backlog.md`
6. `docs/testing-strategy.md`
7. applicable ADRs
8. the current GitHub issue and blockers

## Startup procedure

1. Inspect repository status, open PRs, open issues and CI.
2. Verify `manifests/upstreams.yaml` pins are still reachable.
3. Identify the earliest unblocked issue on the critical path.
4. Check whether independent H0 work can run in parallel.
5. Assign one clear owner per shared subsystem.
6. Create branches named `issue-<n>-<slug>`.
7. Require each worker to return:
   - commits or patch;
   - tests;
   - provenance;
   - risks;
   - remaining blockers;
   - benchmark data when relevant.
8. Integrate in dependency order.
9. Run the full current gate after each merge.
10. Update issue checklists and project docs.

## Decomposition rules

Good worker tasks are bounded by one interface:

- loader/config;
- one quant family;
- CPU backend;
- GPU kernel;
- cache policy;
- TP/ownership;
- telemetry;
- benchmark harness;
- operations.

Do not ask multiple workers to modify the same hot files simultaneously. Do not split a correctness reference from the tests that define it.

## Evidence policy

The orchestrator rejects a result that lacks:

- exact source/base commit;
- reproducible commands;
- actual output;
- reference comparison;
- failure-path test;
- hardware level;
- model/tensor identity where applicable.

Performance results must include raw samples. A screenshot or a single tok/s line is not sufficient.

## Hardware-unavailable mode

Until the P4s arrive:

- prioritize H0 and H1 issues;
- compile `sm_61` code in CUDA 12.6 containers;
- use host simulation and independent references;
- prepare H2 scripts and expected outputs;
- do not mark P4 runtime issues complete;
- do not optimize against unrelated modern GPUs and assume transferability.

When the cards arrive, pause feature expansion and execute the hardware arrival checklist before tuning.

## Integration order

Within a phase:

1. tests and reference path;
2. architecture/data structures;
3. basic implementation;
4. hardware/correctness evidence;
5. optimization;
6. docs and defaults.

An optimized path stays off by default until step 4 and the performance gate pass.

## Review checklist

The orchestrator verifies:

- issue scope is complete but not broadened;
- dependency direction remains acyclic;
- no moving upstream branch is unpinned;
- copied code retains attribution;
- tensor shapes, strides and types are validated;
- fixed-address graph/cache assumptions are explicit;
- cache-zero and pure-CPU modes still work;
- logs expose actual runtime selection;
- tests use the shipping code, not a duplicate reimplementation;
- hardware claims use the target architecture.

## Decision handling

When a material design change is needed:

1. write or amend an ADR;
2. describe alternatives and evidence;
3. update architecture and backlog;
4. obtain owner approval when the change affects v1 scope, license, model artifacts, or release criteria;
5. only then implement.

## Completion protocol

For each issue:

1. all acceptance boxes checked with links;
2. PR merged;
3. CI green;
4. docs/manifests current;
5. dependent issues unblocked;
6. raw artifacts retained when required;
7. issue closed with a concise evidence summary.

For release:

1. freeze upstream pins;
2. run H4 from a clean checkout;
3. publish benchmark artifact bundle;
4. tag release;
5. verify image and source checksums;
6. update known limitations;
7. close the epic only after every required child issue is closed.

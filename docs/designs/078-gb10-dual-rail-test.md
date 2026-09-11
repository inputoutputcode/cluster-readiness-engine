# ADR-078: GB10 Dual-Rail Cluster Test

Status: Proposed for a future PR. The local implementation is on
`gb10support`; this draft does not represent project approval.

## Context

The initial target is a two-node k3s cluster, spark-1ac4 and spark-38fc.
Both nodes advertise NVIDIA-GB10 and one nvidia.com/gpu resource. Both
advertise 63 shared allocation slots under each of
rdma/rdma_shared_device_a and rdma/rdma_shared_device_b.

The supplied plugin logs map resource a to enp1s0f1np1 and resource b to
enP2p1s0f1np1. On spark-38fc these correspond to active Ethernet RDMA
devices rocep1s0f1 and roceP2p1s0f1 on separate IPv4 networks. Equivalent
device mappings and connectivity on spark-1ac4 remain to be checked.
Negotiated link speed and GPU-direct RDMA functionality have not been
established by the supplied output.

CRE parses GB10 labels but its unknown-architecture GPU default is four.
The existing nicResourceName option represents only one resource name;
mlnxPerNode cannot express one allocation from each of two resource pools.
The GB200/GB300 on-prem overrides do not match GB10.

## Decision

Add a reusable multi-resource Certification option and an on-prem GB10 catalog
profile based on the settings validated on the two-node Spark cluster. Keep the
Workflow overlay for single-rail diagnosis. Add a dedicated C2C measurement and
per-node GB10 C2C category, plus a random-initialized Llama 3.2 1B distributed
training category so the recommended suite covers hardware-local coherent
memory, collectives, and end-to-end training goodput.

## Implementation

The local prototype under `tools/gb10/` provides:

1. A Certification render source with gpusPerNode: 1, nodesPerJob: 2,
   enableMNNVL: false, maxBytes: 1G, numIterations: 20, and numCycles: 2.
2. A dependency-free Python overlay that transforms the resolved on-prem
   Workflow. Each worker requests one GPU and one shared allocation slot
   from each selected RDMA resource in both requests and limits.
3. Host networking with ClusterFirstWithHostNet DNS on workers only. The
   launcher uses the Kubernetes pod network so OpenMPI does not mistake the
   same-node worker's host IP for the launcher and run rank 0 outside the
   RDMA-allocated worker container. Worker sshd, readiness probes, and MPI SSH
   arguments consistently use a dedicated port, default 2222. An explicit
   mpirun MCA argument pins rank-to-rank BTL traffic to the selected socket
   interface, avoiding unreachable Docker bridge addresses. The OOB interface
   remains automatic because the launcher uses the CNI network. Runs are
   performed sequentially to avoid host-port conflicts.
4. Exact HCA selection and NCCL_NET=IB, with diagnostic logging and MNNVL
   disabled. NCCL_NET_PLUGIN=none selects NCCL's internal verbs transport
   because the base image's external HPC-X plugin found the selected HCA on
   only one node during the first live test. MPI control traffic uses TCP on
   a configurable bootstrap interface. No fixed GID index or GPU-direct mode
   is forced.
5. Separate rail-a, rail-b, and dual-rail manifests with distinct Workflow
   and runtime names. A node preflight checks labels, readiness, cordon
   state, and advertised resources before rendering.
6. An optional arm64 image recipe building MPI NCCL tests for SM 12.1, plus
   a runbook covering installation, image distribution, server-side dry-run,
   execution, log collection, per-port counters, and cleanup.
7. `make test-gb10`, including tests against the actual catalog renderer and
   a round trip through CRE's typed Workflow decoder. Tests cover resource
   selection, rank counts, networking, SSH ports, images, rejected input,
   and preservation of benchmark arguments.
8. A one-second BandwidthMeasurement sampling interval. The live two-node test
   completed before the catalog's 30-second interval and its launcher pod was
   cleaned up before any bandwidth rows were captured.
9. A `C2CMeasurement` controller and `diagnostics/gb10-c2c` category. The
   benchmark validates managed and mapped pinned memory in both directions and
   reports bandwidth, latency, sample count, and correctness through the CRD,
   Prometheus, thresholds, and `nvcrectl certification report`.
10. A `training/llama32-1b` category using the published Llama 3.2 1B shape,
    random initialization, synthetic tokens, BF16, DDP, and the existing
    Megatron log format for goodput collection without model or dataset access.
11. Include the existing `diagnostics/dcgm-level4` category in the recommended
    suite. A per-category image override retains the DCGM image instead of the
    suite-wide GB10 workload image. It keeps the catalog's one-node grouping,
    two-hour timeout, and external GPU Operator Host Engine dependency.
12. Add the all-gather collective to the GB10 communication profile using the
    same dual-rail runtime resources, MPI transport arguments, and one-second
    bandwidth sampling as all-reduce and all-to-all. Apply the same provisional
    18 GB/s bus-bandwidth floor to all three collectives. Five independent
    all-gather runs measured 20.43-20.79 GB/s (20.52 GB/s median, approximately
    0.7% sample coefficient of variation); observed all-to-all results were
    21.44-21.65 GB/s and all-reduce results were 22.38-23.05 GB/s.
13. A recommended six-category Certification and a separate 4-8-node
    diagnose Certification. Diagnose retains `minGroupSize: 2`; its MNNVL-only
    comparison is skipped because GB10 has NVLink-C2C, not Multi-Node NVLink.

The source Certification is now the primary object to apply. The controller
resolves its GB10 catalog override into the same site-specific worker and MPI
configuration proven by the diagnostic Workflow overlay.

## Rationale

The existing Workflow API can carry the necessary pod and MPI settings. A
small, typed Certification API addition makes distinct device-plugin resource
pools representable without hardcoding their names into the catalog. The GB10
catalog override supplies the platform wiring that the existing generic
on-prem profiles cannot infer.

## Consequences

The Certification path now exercises status aggregation, threshold validation,
and `nvcrectl certification report`. It requires a manager and CRD built from
this branch. The released manager cannot interpret the new API or catalog
profile.

The test requires CRE, Kubeflow Trainer, JobSet, and the GPU/RDMA device
plugins to be installed. Host-networked workers require an available test
SSH port, working inter-node routing, and pod DNS. Node preflight does not
establish free resource capacity, image compatibility, or peer reachability.

Device-plugin advertisement proves allocation availability, not GPU-direct
RDMA or aggregate dual-rail throughput. NCCL transport logs and substantial
payload counter deltas on both ports are required to interpret measurements.
Every category with a CRE measurement has an explicit threshold. DCGM level 4
has no numeric CRE metric; successful completion of the diagnostic process is
its pass criterion.

## Alternatives Considered

- Set mlnxPerNode to two: this cannot request two distinct resource names.
- Extend the GB200/GB300 selectors: this conflates different platform and
  interconnect assumptions.
- Continue with the Workflow-only overlay: this cannot drive Certification
  status aggregation, threshold evaluation, or `nvcrectl certification report`.
- Use TCP only: useful for troubleshooting, but insufficient to validate
  the requested RoCE links.

## Notes

The user authorized local implementation without ADR approval and requested
retaining this draft for a later PR. Project acceptance is still pending.

Local verification passed: generated manifests and deepcopy code, lint, build,
the complete unit/integration suite, and all six GB10 diagnostic tests. The
Certification CRD schema and intentional render/integration goldens changed.
The arm64 workload image and live GPU/RDMA execution remain target-cluster
artifacts and cannot be reproduced on this development host.

Live testing has confirmed MPI launch, pod DNS, SSH on both nodes, both GB10s,
and active 200 Gb/s rail-A HCAs. The external HPC-X NCCL RDMA plugin initialized
the HCA on spark-1ac4 but reported no device on spark-38fc; the internal verbs
transport showed the same asymmetry. Logs then established that rank 0 was
running locally in the host-networked launcher on spark-38fc rather than in its
RDMA-allocated worker. Moving the launcher to the pod network placed both ranks
correctly, after which OpenMPI selected an unreachable Docker bridge address
for rank-to-rank TCP. A worker environment pin did not propagate through the
SSH-launched rank; the equivalent mpirun MCA argument fixed that path. The
resulting runs completed successfully with peak bus bandwidth of 13.93 GB/s on
rail A, 13.89 GB/s on rail B, and 22.36 GB/s with both rails selected. Before a
PR, record the image/driver versions, transport/GDR diagnostics, and per-port
counter deltas. Use that evidence to review the scope and defaults proposed
here.

## References

- [Local GB10 runbook](../../tools/gb10/README.md)
- [ADR-027: Kustomize-like Override UX](027-kustomize-override-ux.md)
- [ADR-075: On-prem GB200/GB300 Override](075-onprem-gb200-gb300-override.md)
- [ADR-077: Certification Workload Image Override](077-workload-image-override.md)
- [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- User-provided node and RDMA plugin output, September 10, 2026

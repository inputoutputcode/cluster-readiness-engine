# ADR-078: GB10 Dual-Rail Cluster Test

Status: Proposed for a future PR. The local prototype is implemented on
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

Propose an opt-in, site-specific two-node NCCL all-reduce test using CRE's
existing Certification renderer and Workflow API. Keep node names, resource
names, network interfaces, image, namespace, runtime class, and SSH port
configurable in the test assets. Establish successful RDMA operation before
selecting general GB10 catalog defaults or changing the CRD schema.

## Implementation

The local prototype under `tools/gb10/` provides:

1. A Certification render source with gpusPerNode: 1, nodesPerJob: 2,
   enableMNNVL: false, maxBytes: 1G, numIterations: 20, and numCycles: 2.
2. A dependency-free Python overlay that transforms the resolved on-prem
   Workflow. Each worker requests one GPU and one shared allocation slot
   from each selected RDMA resource in both requests and limits.
3. Host networking with ClusterFirstWithHostNet DNS. Worker sshd, readiness
   probes, and MPI SSH arguments consistently use a dedicated port, default
   2222. Runs are performed sequentially to avoid host-port conflicts.
4. Exact HCA selection and NCCL_NET=IB, with diagnostic logging and MNNVL
   disabled. MPI control traffic uses TCP on a configurable bootstrap
   interface. No fixed GID index or GPU-direct mode is forced.
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

The transformed Workflow is the object to apply. Applying the source
Certification directly would omit the site-specific network configuration.

## Rationale

The existing Workflow API can carry the necessary pod and MPI settings.
An explicit site test makes the two shared resource pools representable
without widening the public API or hardcoding this site's NIC names into
the default catalog. The reduced message range provides an initial
functional test before longer performance runs.

## Consequences

The first deliverable validates a CRE Workflow and its BandwidthMeasurement,
not end-to-end Certification status aggregation. General GB10 Certification
support remains a subsequent decision informed by hardware results.

The test requires CRE, Kubeflow Trainer, JobSet, and the GPU/RDMA device
plugins to be installed. Host-networked workers require an available test
SSH port, working inter-node routing, and pod DNS. Node preflight does not
establish free resource capacity, image compatibility, or peer reachability.

Device-plugin advertisement proves allocation availability, not GPU-direct
RDMA or aggregate dual-rail throughput. NCCL transport logs and substantial
payload counter deltas on both ports are required to interpret measurements.
No unmeasured performance threshold is supplied.

## Alternatives Considered

- Set mlnxPerNode to two: this cannot request two distinct resource names.
- Extend the GB200/GB300 selectors: this conflates different platform and
  interconnect assumptions.
- Immediately add a multi-resource CRD field: defer until testing confirms
  which runtime settings are required and which should become public API.
- Use TCP only: useful for troubleshooting, but insufficient to validate
  the requested RoCE links.

## Notes

The user authorized local implementation without ADR approval and requested
retaining this draft for a later PR. Project acceptance is still pending.

Local verification passed: lint, build, the unit/integration suite, and all
six GB10 tests. Helm rendering tests were skipped because Helm was unavailable.
The arm64 workload image has not been built here, and live GPU/RDMA tests
remain to be run on the target nodes. No golden files or CRD schemas changed.

Before a PR, record the image/driver versions, mappings on both nodes,
single-rail and dual-rail results, transport/GDR diagnostics, and counter
deltas. Use that evidence to review the scope and defaults proposed here.

## References

- [Local GB10 runbook](../../tools/gb10/README.md)
- [ADR-027: Kustomize-like Override UX](027-kustomize-override-ux.md)
- [ADR-075: On-prem GB200/GB300 Override](075-onprem-gb200-gb300-override.md)
- [ADR-077: Certification Workload Image Override](077-workload-image-override.md)
- [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- User-provided node and RDMA plugin output, September 10, 2026

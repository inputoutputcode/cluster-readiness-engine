# Two-node GB10 RoCE test

This local test renders the existing NCCL all-reduce catalog and applies a
site-specific Workflow overlay. It does not require a new CRE controller or
CRD. It tests a Workflow and its BandwidthMeasurement, not Certification status
aggregation. Do not apply `certification.json` directly: it is input to the
renderer and does not include the RDMA overlay.

The defaults match spark-1ac4 and spark-38fc: one GB10 per node, two shared
RDMA resource pools, and the interface names reported by spark-38fc. Each
worker requests one slot per selected pool; 63 advertised slots are not 63
physical NICs. `mlnxPerNode: 0` leaves the base catalog's NIC requests disabled;
the overlay explicitly supplies the selected resource names.

## Prerequisites

- CRE, its NCCL log profile, Kubeflow Trainer, and JobSet installed and healthy.
- One allocatable NVIDIA GPU on each arm64 node and the RDMA shared plugin.
- Python 3, kubectl pointing to the two-node cluster, and nvcrectl.
- An arm64 image containing CUDA support for GB10, compatible NCCL/verbs
  libraries, `/usr/local/bin/all_reduce_perf_mpi`, `/usr/local/mpi/bin/mpirun`,
  SSH server/client, and `ibv_devinfo`.

Build nvcrectl from the repository if necessary:

```bash
go build -o bin/nvcrectl ./cmd/nvcrectl/
```

Check the installation with `./bin/nvcrectl setup status`. If components are
missing, follow [cluster installation](../../docs/getting-started/install.md).
`nvcrectl setup init` installs CRE and its dependencies; when using a CLI built
from this branch, pass `--version` with a published chart version, because the
local development build has no matching published chart. This overlay uses
existing APIs, so no locally built controller image is needed.

The optional Dockerfile builds the MPI NCCL test binary for SM 12.1 using the
catalog's PyTorch base image. Build on a GB10 node (or an arm64 builder):

```bash
docker build -t gb10-nccl:local tools/gb10
docker save gb10-nccl:local -o /tmp/gb10-nccl.tar
sudo k3s ctr images import /tmp/gb10-nccl.tar
scp /tmp/gb10-nccl.tar spark-1ac4:/tmp/gb10-nccl.tar
ssh spark-1ac4 'sudo k3s ctr images import /tmp/gb10-nccl.tar'
```

This example assumes the build runs on spark-38fc. Alternatively use a registry
accessible by both nodes. Supply `--pull-secret NAME` if required. The image
recipe still needs a real arm64 build and a GPU run on your machines; rendering
the manifest cannot verify driver compatibility. Base image, NCCL test tag,
MPI_HOME, and NCCL_HOME are Docker build arguments if your installation differs.

## Check the nodes

The earlier DaemonSet lookup contained spelling errors. This displays its
actual configuration sources, including ConfigMaps and hostPath mounts:

```bash
kubectl -n kube-system get ds rdma-shared-dp-ds -o json | jq '.spec.template.spec.volumes'
kubectl get runtimeclass
kubectl get nodes -o wide
```

On **both** nodes run:

```bash
rdma link show
ip -br address
sudo ethtool enp1s0f1np1
sudo ethtool enP2p1s0f1np1
ss -ltn 'sport = :2222'
```

Confirm `rocep1s0f1` and `roceP2p1s0f1` are ACTIVE and the test SSH port is
unused. Verify each rail can reach its peer's address using `ping -I INTERFACE
PEER_ADDRESS`. The supplied output only confirmed local link state on
spark-38fc, not peer reachability or negotiated speed. Ensure Kubernetes pod
DNS and the test SSH port are reachable between the nodes.

## Render and run

Run **one case at a time**. The runs share GPUs and host SSH port 2222.
Start with rail A, then B, then both. Each has a distinct Workflow/runtime name.

```bash
python3 tools/gb10/render.py --image gb10-nccl:local --rail a --preflight > /tmp/gb10-a.json
./bin/nvcrectl workflow render /tmp/gb10-a.json --dry-run
kubectl apply --dry-run=server -f /tmp/gb10-a.json
kubectl apply -f /tmp/gb10-a.json
kubectl get workflows.nvcre.nvidia.com gb10-a -w
```

If the NVIDIA runtime is not k3s's default, add `--runtime-class nvidia` to
the render command, using the actual RuntimeClass name shown above. Custom
node names, resource names, HCAs, namespace, bootstrap interface and SSH port
are flags; see `python3 tools/gb10/render.py --help`. The default bootstrap
interface stays on rail A even in the rail-B test; NCCL data is restricted to
the selected HCA. Override `--socket-ifname enP2p1s0f1np1` for bootstrap on B.

`--preflight` checks node identity, readiness and advertised resources. It does
not check current resource consumption, peer links, image contents, or port
availability. Host networking exposes the node's interfaces; device allocation
exposes the selected verbs devices. MPI uses TCP for control; NCCL is explicitly
set to `NET=IB` for RoCE data and cannot silently substitute its Socket backend.
`NCCL_NET_PLUGIN=none` bypasses the HPC-X `libnccl-net.so` included in the base
image and selects NCCL's internal IB verbs transport. This avoids an observed
case where that external plugin found the selected HCA on one GB10 but reported
`NET/IB : No device found` on the other.

Collect status and logs before deleting the run:

```bash
kubectl get workflows.nvcre.nvidia.com gb10-a -o yaml
kubectl get jobs.nvcre.nvidia.com,trainjobs,bandwidthmeasurements.nvcre.nvidia.com
kubectl get pods -o wide
kubectl logs POD_NAME -c node
kubectl get bandwidthmeasurements.nvcre.nvidia.com -o yaml
kubectl delete -f /tmp/gb10-a.json --wait=true
```

Use the launcher's pod name for NCCL results; worker logs contain device and
GPU startup checks. Wait for the run's pods to disappear and port 2222 to be
released before repeating the commands with `--rail b` and then `--rail both`
(and correspondingly named output files and Workflow names).

## Interpret the result

The workload uses two ranks and one GPU per rank, sweeps 8 bytes through 1 GiB,
and runs 20 iterations with two cycles. Confirm two GB10 ranks, no correctness
errors, `NCCL_NET_PLUGIN set by environment to none`, and internal NET/IB device
selection on both ranks in the launcher log. CRE records algBW and busBW;
there is deliberately no unmeasured pass/fail bandwidth threshold.

Capture port counters on both nodes immediately before and after each run:

```bash
for dev in rocep1s0f1 roceP2p1s0f1; do
  for counter in port_xmit_data port_rcv_data; do
    printf '%s %s: ' "$dev" "$counter"
    cat "/sys/class/infiniband/$dev/ports/1/counters/$counter"
  done
done
```

Compare deltas, avoiding concurrent network workloads. Both selected ports
must show substantial payload traffic above bootstrap/control traffic to
claim dual-rail use; selecting both HCAs alone does
not prove NCCL used both. RDMA transport success does not by itself prove
GPU-direct RDMA: inspect NCCL's GDR diagnostics separately. Do not assume a
twofold bandwidth improvement or derive a threshold from nominal link speed.

## Local tests

```bash
make test-gb10
# Or run just the Python tests with an existing CLI:
python3 -m unittest discover -s tools/gb10 -v
NVCRECTL="$PWD/bin/nvcrectl" python3 -m unittest discover -s tools/gb10 -v
```

The second command also runs all three overlays against the actual catalog
renderer, checking rank counts, selected resources, SSH port consistency,
images, host networking, and preservation of NCCL measurement arguments.
No golden files need regeneration.

References: [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests) and
[NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

# GB10 cluster certification suite

This directory contains the recommended Certification path for a 2-8 node DGX
Spark cluster and a lower-level Workflow overlay for single-rail diagnostics. The
Certification path requires the CRD and manager built from `gb10support`; the
released v0.2.0 manager does not contain the GB10 catalog profile.

The defaults match spark-1ac4 and spark-38fc: one GB10 per node, two shared
RDMA resource pools, and the interface names reported by spark-38fc. Each
worker requests one slot per selected pool; 63 advertised slots are not 63
physical NICs. The Certification's `nicResources` list supplies both resource
names explicitly. The diagnostic overlay selects one or both entries from the
same list.

## Run a Certification

Install the generated Certification CRD and deploy a manager image built from
this branch before using this path. The catalog is embedded in the manager, so
installing only the branch CLI while leaving the v0.2.0 manager running is not
enough. On spark-38fc, build and load the manager image on both k3s nodes, then
upgrade the existing Helm release from the local chart:

```bash
make docker-build IMG=nvcre-manager:gb10
docker save nvcre-manager:gb10 -o /tmp/nvcre-manager-gb10.tar
sudo k3s ctr images import /tmp/nvcre-manager-gb10.tar
scp /tmp/nvcre-manager-gb10.tar spark-1ac4:/tmp/
ssh -t spark-1ac4 'sudo k3s ctr images import /tmp/nvcre-manager-gb10.tar'

kubectl apply -f helm/cluster-readiness-engine/crds/nvcre.nvidia.com_certifications.yaml
kubectl apply -f helm/cluster-readiness-engine/crds/nvcre.nvidia.com_jobs.yaml
kubectl apply -f helm/cluster-readiness-engine/crds/nvcre.nvidia.com_workflows.yaml
kubectl apply -f helm/cluster-readiness-engine/crds/nvcre.nvidia.com_c2cmeasurements.yaml
helm upgrade --install nvcre helm/cluster-readiness-engine \
  --namespace nvcre \
  --create-namespace \
  --set manager.image.repository=nvcre-manager \
  --set manager.image.tag=gb10 \
  --set manager.image.pullPolicy=Never \
  --set metrics.serviceMonitor.enabled=false
kubectl -n nvcre rollout status deployment/nvcre-manager
```

The `-t` on the remote SSH command allocates the terminal required by `sudo`.
If the deployment has a different name, find it with `kubectl -n nvcre get
deployments`. Build the matching CLI, preview the exact resources against the
live cluster, and run the Certification:

```bash
make build
go build -o bin/nvcrectl ./cmd/nvcrectl/

./bin/nvcrectl certification render \
  --platform onprem \
  --dry-run \
  tools/gb10/certification.json

./bin/nvcrectl certification run \
  --cert-file tools/gb10/certification.json \
  --wait \
  --results-file /tmp/gb10-certification.json
```

The recommended sample runs five categories: per-node DCGM level-4 diagnostics,
per-node C2C coherent-memory validation, dual-rail NCCL all-reduce, dual-rail
NCCL all-to-all, and a random-initialized Llama 3.2 1B DDP training run. The
DCGM category retains its `nvcr.io/nvidia/cloud-native/dcgm` image instead of
inheriting the suite-wide GB10 workload image. It enforces the provisional
dual-rail threshold observed during local validation (`busBandwidthGBps >= 18`),
a C2C sanity floor of 1 GB/s in each direction, and runtime goodput of at least
0.80. The lower goodput floor accounts for fixed model and DDP startup overhead
in the short 50-step smoke run. Establish site baselines before tightening the
C2C or training thresholds.
Leave the Certification installed to regenerate its report later:

```bash
./bin/nvcrectl certification report gb10-cert \
  -n default \
  --results-file /tmp/gb10-certification.json
```

## Prerequisites

- CRE, its NCCL log profile, Kubeflow Trainer, and JobSet installed and healthy.
- A reachable DCGM Host Engine at `nvidia-dcgm.gpu-operator.svc:5555`. The
  existing `diagnostics/dcgm-level4` catalog category is a DCGM client and does
  not deploy Host Engine. GPU Operator commonly disables the standalone DCGM
  service unless it is explicitly enabled.
- One allocatable NVIDIA GPU on each arm64 node and the RDMA shared plugin.
- Python 3, kubectl pointing to the two-node cluster, and nvcrectl.
- An arm64 image containing CUDA support for GB10, compatible NCCL/verbs
  libraries, `/usr/local/bin/all_reduce_perf_mpi`,
  `/usr/local/bin/alltoall_perf_mpi`, `/usr/local/bin/gb10-c2c`, the synthetic
  Llama workload, `/usr/local/mpi/bin/mpirun`, SSH server/client, and
  `ibv_devinfo`.

Build nvcrectl from the repository if necessary:

```bash
go build -o bin/nvcrectl ./cmd/nvcrectl/
```

Check the installation with `./bin/nvcrectl setup status`. If components are
missing, follow [cluster installation](../../docs/getting-started/install.md).
`nvcrectl setup init` installs CRE and its dependencies; when using a CLI built
from this branch, pass `--version` with a published chart version, because the
local development build has no matching published chart. Then apply the branch
CRD and local manager upgrade shown above. Re-running `setup init` by itself
would restore the released manager, which does not understand `nicResources`.

Verify the DCGM dependency before starting the full suite:

```bash
kubectl -n gpu-operator get service nvidia-dcgm
kubectl -n gpu-operator get endpointslice \
  -l kubernetes.io/service-name=nvidia-dcgm
kubectl -n gpu-operator get pods -l app=nvidia-dcgm -o wide
```

The catalog gives each per-node level-4 Job a two-hour timeout. The two nodes
run concurrently unless `options.maxConcurrent` is set. DCGM chooses supported
plugins from its packaged SKU configuration; inspect the completed pod logs for
`Skip` or unsupported-test messages. A skipped diagnostic does not establish
health, even when the process exits successfully. NVIDIA documents higher-level
diagnostics on non-datacenter GPUs as product-dependent, so GB10 support must be
confirmed using the exact DCGM image and driver deployed on the Sparks.

The optional Dockerfile builds the MPI NCCL test binary for SM 12.1 using the
catalog's PyTorch base image. It also builds the C2C CUDA benchmark and installs
the synthetic Llama training script. Build on a GB10 node (or an arm64 builder):

```bash
docker build -t gb10-nccl:local tools/gb10
docker save gb10-nccl:local -o /tmp/gb10-nccl.tar
sudo k3s ctr images import /tmp/gb10-nccl.tar
scp /tmp/gb10-nccl.tar spark-1ac4:/tmp/gb10-nccl.tar
ssh -t spark-1ac4 'sudo k3s ctr images import /tmp/gb10-nccl.tar'
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

## Run single-rail diagnostics

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
availability. Host networking exposes the node's interfaces in the two worker
pods; device allocation exposes the selected verbs devices. The launcher stays
on the Kubernetes pod network. If it shares host networking with a worker,
OpenMPI recognizes the worker's host IP as local and runs rank 0 in the launcher
container, where no RDMA resource was allocated. MPI uses TCP for control and
SSHes into both workers. The `mpirun` MCA arguments pin rank-to-rank BTL TCP to
the selected `--socket-ifname`; a worker container environment setting is not
reliably propagated through SSH. Without the explicit MCA argument, OpenMPI
selected a Docker bridge address (`172.18.0.1`) that was unreachable from the
peer. The OOB interface remains automatic because the launcher uses the CNI
network and does not have the workers' host interface. NCCL is explicitly
set to `NET=IB` for RoCE data and cannot silently substitute its Socket backend.
`NCCL_NET_PLUGIN=none` bypasses the HPC-X `libnccl-net.so` included in the base
image and selects NCCL's internal IB verbs transport. This avoids an observed
case where that external plugin found the selected HCA on one GB10 but reported
`NET/IB : No device found` on the other.

Collect status and logs before deleting the run:

```bash
kubectl get workflows.nvcre.nvidia.com gb10-a -o yaml
kubectl get jobs.nvcre.nvidia.com,trainjobs
kubectl get bandwidthmeasurements.nvcre.nvidia.com,c2cmeasurements.nvcre.nvidia.com
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
and runs 20 iterations with two cycles. The GB10 overlay samples launcher logs
every second because this small two-node test can finish before the catalog's
30-second sampling interval. Confirm two GB10 ranks, no correctness
errors, `NCCL_NET_PLUGIN set by environment to none`, and internal NET/IB device
selection on both ranks in the launcher log. CRE records algBW and busBW. The
sample Certification uses the provisional dual-rail threshold established by
the successful test, 18 GB/s bus bandwidth.

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
images, host networking, C2C and Llama measurement configuration, and
preservation of NCCL measurement arguments.

## Diagnose 4-8 nodes

`testScale: diagnose` is useful once at least four GB10 nodes are available.
The supplied configuration selects all nodes labeled `NVIDIA-GB10`, clamps its
eight-node maximum to available capacity, and keeps bisection groups at two or
more nodes so NCCL remains a meaningful oracle with one GPU per node:

```bash
./bin/nvcrectl certification render \
  --platform onprem \
  --dry-run \
  tools/gb10/certification-diagnose.json

./bin/nvcrectl certification run \
  --cert-file tools/gb10/certification-diagnose.json \
  --wait \
  --timeout 2h \
  --results-file /tmp/gb10-diagnose.json
```

MNNVL remains disabled. Diagnose still performs screening, bisection,
confirmation, and cross-boundary isolation; only its Multi-Node NVLink
comparison stage is inapplicable to GB10.

References: [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests) and
[NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a site-specific GB10 Workflow without modifying the CRE catalog."""

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import sys


def named(items, name):
    matches = [item for item in items if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r}, found {len(matches)}")
    return matches[0]


def configure(workflows, args):
    matches = [workflow for workflow in workflows
               if workflow.get("kind") == "Workflow"
               and workflow.get("spec", {}).get("jobTemplate", {}).get("spec", {})
                   .get("bandwidthMeasurement", {}).get("testType") == "all_reduce"]
    if len(matches) != 1:
        raise ValueError(f"expected one rendered all-reduce Workflow, found {len(matches)}")
    wf = copy.deepcopy(matches[0])
    spec = wf["spec"]
    if spec.get("overrides"):
        raise ValueError("render platform overrides before applying the GB10 overlay")
    tj = spec["jobTemplate"]["spec"]["workload"]["trainJob"]
    spec["jobTemplate"]["spec"]["bandwidthMeasurement"]["sampleInterval"] = "1s"
    trainer = tj["trainer"]
    binary = "/usr/local/bin/all_reduce_perf_mpi"
    old_args = trainer["args"]
    if binary not in old_args:
        raise ValueError("expected NCCL all-reduce catalog arguments")
    perf_args = old_args[old_args.index(binary):]
    runtimes = [d for d in spec["dependencies"] if d.get("kind") == "TrainingRuntime"]
    if len(runtimes) != 1 or len(spec["dependencies"]) != 1:
        raise ValueError("expected only the base on-prem NCCL TrainingRuntime")
    rt = runtimes[0]
    run_name = f"gb10-{args.rail}"
    wf["metadata"] = {"name": run_name, "namespace": args.namespace}
    rt["metadata"]["name"] = f"{run_name}-runtime"
    rt["metadata"]["namespace"] = args.namespace
    tj["runtimeRef"]["name"] = rt["metadata"]["name"]
    rt["spec"]["mlPolicy"]["mpi"]["numProcPerNode"] = 1
    trainer["numNodes"] = 2
    trainer["numProcPerNode"] = 1
    trainer["image"] = args.image
    selected = []
    if args.rail in ("a", "both"):
        selected.append((args.resource_a, args.hca_a))
    if args.rail in ("b", "both"):
        selected.append((args.resource_b, args.hca_b))
    trainer["command"] = ["timeout", "1800", "/usr/local/mpi/bin/mpirun"]
    trainer["args"] = [
        "-N", "1", "--allow-run-as-root", "--bind-to", "none",
        "--mca", "plm_rsh_args", f"-p {args.ssh_port} -o StrictHostKeyChecking=no",
        "--mca", "pml", "ob1", "--mca", "btl", "self,tcp",
        "--mca", "btl_tcp_if_include", args.socket_ifname,
    ]
    env = {
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
        "NCCL_MNNVL_ENABLE": "0",
        "NCCL_NET": "IB",
        # The PyTorch image ships HPC-X libnccl-net.so. Use NCCL's internal
        # verbs transport so both GB10 nodes follow the same device path.
        "NCCL_NET_PLUGIN": "none",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": "=" + ",".join(hca + ":1" for _, hca in selected),
        "NCCL_SOCKET_IFNAME": "=" + args.socket_ifname,
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_CROSS_NIC": "0",
    }
    for key, value in env.items():
        trainer["args"].extend(["-x", f"{key}={value}"])
    trainer["args"].extend(perf_args)
    orch = spec["orchestration"]
    orch["target"] = {"nodeNames": args.nodes}
    orch.pop("topology", None)
    orch.pop("diagnose", None)
    orch["execution"] = {"maxConcurrent": 1, "timeoutPerJob": "35m"}
    orch["iterations"] = 1
    jobs = rt["spec"]["template"]["spec"]["replicatedJobs"]
    worker = named(jobs, "node")["template"]["spec"]["template"]["spec"]
    for job in jobs:
        pod = job["template"]["spec"]["template"]["spec"]
        if job["name"] == "node":
            pod["hostNetwork"] = True
            pod["dnsPolicy"] = "ClusterFirstWithHostNet"
        else:
            # A host-networked launcher on the same node as a worker makes
            # OpenMPI classify that worker as local and execute rank 0 in the
            # launcher, which has no GPU/RDMA allocation. Keep the launcher on
            # the pod network so both ranks are started through worker sshd.
            pod.pop("hostNetwork", None)
            pod.pop("dnsPolicy", None)
        pod["nodeSelector"] = {"nvidia.com/gpu.product": "NVIDIA-GB10", "kubernetes.io/arch": "arm64"}
        if args.runtime_class:
            pod["runtimeClassName"] = args.runtime_class
        if args.pull_secret:
            pod["imagePullSecrets"] = [{"name": args.pull_secret}]
        pod["tolerations"] = [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
            {"key": "kubernetes.io/arch", "operator": "Equal", "value": "arm64", "effect": "NoSchedule"},
        ]
        for container in pod["containers"] + pod.get("initContainers", []):
            container["image"] = args.image
            container["imagePullPolicy"] = "IfNotPresent"
    container = named(worker["containers"], "node")
    resources = {"nvidia.com/gpu": "1", **{resource: "1" for resource, _ in selected}}
    container["resources"] = {"requests": resources.copy(), "limits": resources.copy()}
    # A dedicated port avoids colliding with the host's administrative sshd.
    container["command"] = ["sh", "-ec"]
    container["args"] = [
        "test -x /usr/local/bin/all_reduce_perf_mpi\n"
        "test -x /usr/local/mpi/bin/mpirun\n"
        "test -c /dev/infiniband/rdma_cm\n"
        "ibv_devinfo\n"
        "nvidia-smi -L\n"
        "mkdir -p /var/run/sshd /root/.ssh\n"
        "chmod 700 /root/.ssh\n"
        "cp /tmp/mpi-ssh-raw/* /root/.ssh/\n"
        "chmod 600 /root/.ssh/id_rsa\n"
        "chmod 644 /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys\n"
        "ssh-keygen -A\n"
        f"exec /usr/sbin/sshd -D -e -p {args.ssh_port} "
        "-o PasswordAuthentication=no -o PermitRootLogin=prohibit-password "
        "-o PidFile=/tmp/gb10-sshd.pid\n"
    ]
    container["readinessProbe"] = {
        "initialDelaySeconds": 5, "tcpSocket": {"port": args.ssh_port}
    }
    return wf


def preflight(nodes, args):
    by_name = {node["metadata"]["name"]: node for node in nodes["items"]}
    for name in args.nodes:
        if name not in by_name:
            raise ValueError(f"node {name} not found")
        node = by_name[name]
        labels = node["metadata"].get("labels", {})
        if labels.get("nvidia.com/gpu.product") != "NVIDIA-GB10":
            raise ValueError(f"{name}: expected NVIDIA-GB10 label")
        if labels.get("kubernetes.io/arch") != "arm64":
            raise ValueError(f"{name}: expected arm64")
        if node.get("spec", {}).get("unschedulable"):
            raise ValueError(f"{name}: node is cordoned")
        if not any(c["type"] == "Ready" and c["status"] == "True"
                   for c in node["status"].get("conditions", [])):
            raise ValueError(f"{name}: node is not Ready")
        required = ["nvidia.com/gpu"]
        if args.rail in ("a", "both"):
            required.append(args.resource_a)
        if args.rail in ("b", "both"):
            required.append(args.resource_b)
        for resource in required:
            if int(node["status"].get("allocatable", {}).get(resource, "0")) < 1:
                raise ValueError(f"{name}: missing allocatable {resource}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, help="existing certification render --output json result")
    result.add_argument("--nvcrectl", default="./bin/nvcrectl")
    result.add_argument("--image", required=True, help="arm64 image with MPI NCCL tests and SSH")
    result.add_argument("--nodes", nargs=2, default=["spark-1ac4", "spark-38fc"])
    result.add_argument("--namespace", default="default")
    result.add_argument("--rail", choices=["a", "b", "both"], default="both")
    result.add_argument("--resource-a", default="rdma/rdma_shared_device_a")
    result.add_argument("--resource-b", default="rdma/rdma_shared_device_b")
    result.add_argument("--hca-a", default="rocep1s0f1")
    result.add_argument("--hca-b", default="roceP2p1s0f1")
    result.add_argument("--socket-ifname", default="enp1s0f1np1")
    result.add_argument("--ssh-port", type=int, default=2222)
    result.add_argument("--runtime-class", default="", help="e.g. nvidia if k3s needs an explicit RuntimeClass")
    result.add_argument("--pull-secret", default="")
    result.add_argument("--preflight", action="store_true", help="read node readiness and advertised resources using kubectl")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.nodes[0] == args.nodes[1]:
        cli.error("two distinct nodes are required")
    if not 1024 <= args.ssh_port <= 65535:
        cli.error("SSH port must be between 1024 and 65535")
    if args.resource_a == args.resource_b or args.hca_a == args.hca_b:
        cli.error("rails must have distinct resource and HCA names")
    for value in [args.socket_ifname, args.hca_a, args.hca_b]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            cli.error("interface/HCA names must contain only letters, digits, _, ., or -")
    try:
        if args.preflight:
            preflight(json.loads(subprocess.check_output(
                ["kubectl", "get", "nodes", "-o", "json"], text=True)), args)
        if args.input:
            workflows = json.loads(args.input.read_text())
        else:
            workflows = json.loads(subprocess.check_output([
                args.nvcrectl, "certification", "render", "--platform", "onprem",
                "--output", "json", str(Path(__file__).with_name("certification.json")),
            ], text=True))
        print(json.dumps(configure(workflows, args), indent=2))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"GB10 render failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

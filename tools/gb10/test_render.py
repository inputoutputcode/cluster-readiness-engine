# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests; set NVCRECTL to also test the actual catalog render."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import render


def base_workflow():
    def job(name):
        return {"name": name, "template": {"spec": {"template": {"spec": {
            "containers": [{"name": "node", "image": "base", "volumeMounts": [
                {"name": "mpi-ssh-auth", "mountPath": "/tmp/mpi-ssh-raw"}
            ]}],
            "initContainers": [{"name": "ssh-init", "image": "base"}]
        }}}}}
    return [{"apiVersion": "nvcre.nvidia.com/v1alpha1", "kind": "Workflow",
             "metadata": {"name": "original"}, "spec": {
                 "dependencies": [{"apiVersion": "trainer.kubeflow.org/v1alpha1",
                                   "kind": "TrainingRuntime", "metadata": {"name": "runtime"},
                                   "spec": {"mlPolicy": {"mpi": {"numProcPerNode": 4}},
                                            "template": {"spec": {"replicatedJobs": [job("node"), job("launcher")]}}}}],
                 "jobTemplate": {"spec": {"bandwidthMeasurement": {"testType": "all_reduce"},
                                          "workload": {"trainJob": {
                                              "runtimeRef": {"name": "runtime"},
                                              "runtimePatches": [{"manager": "preserve-me"}],
                                              "trainer": {"args": ["-N", "4", "/usr/local/bin/all_reduce_perf_mpi",
                                                                   "-b", "8", "-e", "1G", "-f", "2", "-n", "20", "-N", "2"]}
                                          }}}},
                 "orchestration": {"target": {}, "iterations": 1}
             }}]


class GB10Test(unittest.TestCase):
    def options(self, *extra):
        return render.parser().parse_args(["--image", "local/gb10:test", *extra])

    def assert_contract(self, wf, rail):
        spec = wf["spec"]
        self.assertEqual(spec["jobTemplate"]["spec"]["bandwidthMeasurement"]["sampleInterval"], "1s")
        tj = spec["jobTemplate"]["spec"]["workload"]["trainJob"]
        trainer = tj["trainer"]
        self.assertEqual((trainer["numNodes"], trainer["numProcPerNode"]), (2, 1))
        self.assertIn("NCCL_MNNVL_ENABLE=0", trainer["args"])
        self.assertIn("NCCL_NET=IB", trainer["args"])
        self.assertIn("NCCL_NET_PLUGIN=none", trainer["args"])
        hcas = {"a": "rocep1s0f1:1", "b": "roceP2p1s0f1:1", "both": "rocep1s0f1:1,roceP2p1s0f1:1"}
        self.assertIn("NCCL_IB_HCA==" + hcas[rail], trainer["args"])
        self.assertEqual(trainer["args"][-10:], ["-b", "8", "-e", "1G", "-f", "2", "-n", "20", "-N", "2"])
        self.assertEqual(spec["orchestration"]["target"]["nodeNames"], ["spark-1ac4", "spark-38fc"])
        rt = spec["dependencies"][0]
        self.assertEqual(tj["runtimeRef"]["name"], rt["metadata"]["name"])
        self.assertEqual(len(spec["dependencies"]), 1)
        self.assertTrue(tj["runtimePatches"])
        for job in rt["spec"]["template"]["spec"]["replicatedJobs"]:
            pod = job["template"]["spec"]["template"]["spec"]
            if job["name"] == "node":
                self.assertTrue(pod["hostNetwork"])
                self.assertEqual(pod["dnsPolicy"], "ClusterFirstWithHostNet")
            else:
                self.assertNotIn("hostNetwork", pod)
                self.assertNotIn("dnsPolicy", pod)
            for container in pod["containers"] + pod.get("initContainers", []):
                self.assertEqual(container["image"], "local/gb10:test")
            if job["name"] == "node":
                worker = render.named(pod["containers"], "node")
                resources = {"nvidia.com/gpu": "1"}
                for letter in (["a", "b"] if rail == "both" else [rail]):
                    resources[f"rdma/rdma_shared_device_{letter}"] = "1"
                self.assertEqual(worker["resources"], {"requests": resources, "limits": resources})
                self.assertEqual(worker["readinessProbe"]["tcpSocket"]["port"], 2222)
                self.assertIn("-p 2222", worker["args"][0])
                self.assertTrue(worker["volumeMounts"])
        btl_index = trainer["args"].index("btl_tcp_if_include")
        self.assertEqual(trainer["args"][btl_index + 1], "enp1s0f1np1")
        self.assertNotIn("oob_tcp_if_include", trainer["args"])

    def test_all_rails_and_no_input_mutation(self):
        base = base_workflow()
        original = copy.deepcopy(base)
        for rail in ["a", "b", "both"]:
            with self.subTest(rail=rail):
                self.assert_contract(render.configure(base, self.options("--rail", rail)), rail)
        self.assertEqual(base, original)

    def test_custom_interfaces_port_and_runtime(self):
        args = self.options("--hca-a", "custom0", "--hca-b", "custom1", "--ssh-port", "2223",
                            "--socket-ifname", "eth9", "--resource-a", "rdma/x", "--resource-b", "rdma/y",
                            "--runtime-class", "nvidia", "--pull-secret", "registry")
        wf = render.configure(base_workflow(), args)
        trainer = wf["spec"]["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]
        self.assertIn("NCCL_IB_HCA==custom0:1,custom1:1", trainer["args"])
        self.assertIn("NCCL_SOCKET_IFNAME==eth9", trainer["args"])
        self.assertIn("-p 2223 -o StrictHostKeyChecking=no", trainer["args"])
        pod = wf["spec"]["dependencies"][0]["spec"]["template"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
        self.assertEqual(pod["runtimeClassName"], "nvidia")
        self.assertEqual(pod["imagePullSecrets"], [{"name": "registry"}])
        self.assertEqual(pod["containers"][0]["resources"]["limits"]["rdma/y"], "1")
        btl_index = trainer["args"].index("btl_tcp_if_include")
        self.assertEqual(trainer["args"][btl_index + 1], "eth9")

    def test_reject_unresolved_and_wrong_catalog(self):
        base = base_workflow()
        base[0]["spec"]["overrides"] = [{"when": {}}]
        with self.assertRaisesRegex(ValueError, "render platform"):
            render.configure(base, self.options())
        base = base_workflow()
        base[0]["spec"]["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]["args"] = []
        with self.assertRaisesRegex(ValueError, "all-reduce"):
            render.configure(base, self.options())

    def test_node_preflight(self):
        args = self.options()
        nodes = {"items": [{"metadata": {"name": name, "labels": {
            "nvidia.com/gpu.product": "NVIDIA-GB10", "kubernetes.io/arch": "arm64"}},
            "status": {"conditions": [{"type": "Ready", "status": "True"}],
                       "allocatable": {"nvidia.com/gpu": "1", args.resource_a: "63", args.resource_b: "63"}}}
                           for name in args.nodes]}
        render.preflight(nodes, args)
        nodes["items"][1]["status"]["allocatable"].pop(args.resource_b)
        with self.assertRaisesRegex(ValueError, "spark-38fc: missing"):
            render.preflight(nodes, args)
        args.rail = "a"
        render.preflight(nodes, args)
        nodes["items"][1]["status"]["conditions"][0]["status"] = "False"
        with self.assertRaisesRegex(ValueError, "not Ready"):
            render.preflight(nodes, args)

    def test_invalid_cli_arguments_emit_no_manifest(self):
        for extra in [["--nodes", "same", "same"], ["--ssh-port", "22"],
                      ["--socket-ifname", "eth0;false"], ["--resource-b", "rdma/rdma_shared_device_a"]]:
            with self.subTest(extra=extra):
                result = subprocess.run([sys.executable, str(Path(render.__file__)), "--image", "test", *extra],
                                        capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    @unittest.skipUnless(os.environ.get("NVCRECTL"), "set NVCRECTL for actual catalog integration")
    def test_actual_catalog(self):
        base = json.loads(subprocess.check_output([
            os.environ["NVCRECTL"], "certification", "render", "--platform", "onprem", "--output", "json",
            str(Path(__file__).with_name("certification.json"))], text=True))
        self.assertEqual(len(base), 6)
        by_variant = {wf["metadata"]["labels"]["nvcre.nvidia.com/category-variant"]: wf
                      for wf in base}
        dcgm = by_variant["dcgm-level4"]["spec"]
        dcgm_trainer = dcgm["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]
        self.assertEqual(dcgm_trainer["image"],
                         "nvcr.io/nvidia/cloud-native/dcgm:4.5.2-1-ubuntu22.04")
        self.assertEqual(dcgm_trainer["numNodes"], 1)
        self.assertEqual(dcgm["orchestration"]["execution"]["timeoutPerJob"], "2h0m0s")
        dcgm_runtime = dcgm["dependencies"][0]
        dcgm_container = dcgm_runtime["spec"]["template"]["spec"]["replicatedJobs"][0][
            "template"]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(dcgm_container["image"], dcgm_trainer["image"])
        self.assertIn("nvidia-dcgm.gpu-operator.svc:5555", dcgm_trainer["args"])
        c2c = by_variant["gb10-c2c"]["spec"]
        self.assertEqual(c2c["jobTemplate"]["spec"]["c2cMeasurement"]["sampleInterval"], "1s")
        self.assertEqual(c2c["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]["numNodes"], 1)
        self.assertEqual(c2c["orchestration"]["execution"]["maxConcurrent"], 1)
        self.assertEqual(c2c["validation"]["performance"]["thresholds"]["thresholds"]["c2cVerified"],
                         "value >= 1")
        llama = by_variant["llama32-1b"]["spec"]
        llama_trainer = llama["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]
        self.assertEqual(llama_trainer["numNodes"], 2)
        self.assertEqual(llama_trainer["numProcPerNode"], 1)
        self.assertEqual(llama_trainer["command"], ["/bin/bash", "-ceu"])
        for argument in ["torchrun", "PET_NNODES", "PET_NPROC_PER_NODE", "PET_NODE_RANK",
                         "--master-addr", "PET_MASTER_ADDR", "--master-port", "PET_MASTER_PORT"]:
            self.assertIn(argument, llama_trainer["args"][0])
        self.assertNotIn("--rdzv-", llama_trainer["args"][0])
        self.assertEqual(llama["validation"]["performance"]["thresholds"]["thresholds"]["goodputRatio"],
                         "value >= 0.80")
        self.assertEqual(llama["jobTemplate"]["spec"]["goodputMeasurement"]["logProfileRef"],
                         "megatron-training")
        for variant in ["nccl-all-reduce", "nccl-all-gather", "nccl-alltoall"]:
            with self.subTest(variant=variant):
                thresholds = by_variant[variant]["spec"]["validation"]["performance"][
                    "thresholds"]["thresholds"]
                self.assertEqual(thresholds["busBandwidthGBps"], "value >= 18")
        allgather = by_variant["nccl-all-gather"]["spec"]
        self.assertEqual(allgather["jobTemplate"]["spec"]["bandwidthMeasurement"], {
            "logProfileRef": "nccl-bandwidth", "sampleInterval": "1s", "testType": "all_gather"})
        allgather_args = allgather["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]["args"]
        self.assertIn("/usr/local/bin/all_gather_perf_mpi", allgather_args)
        self.assertIn("NCCL_IB_HCA==rocep1s0f1:1,roceP2p1s0f1:1", allgather_args)
        alltoall_args = by_variant["nccl-alltoall"]["spec"]["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]["args"]
        self.assertIn("/usr/local/bin/alltoall_perf_mpi", alltoall_args)
        self.assertIn("NCCL_IB_HCA==rocep1s0f1:1,roceP2p1s0f1:1", alltoall_args)
        for rail in ["a", "b", "both"]:
            with self.subTest(rail=rail):
                wf = render.configure(base, self.options("--rail", rail))
                self.assert_contract(wf, rail)
                # Read the result back through CRE's typed Workflow decoder.
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "workflow.json"
                    path.write_text(json.dumps(wf))
                    nodes = Path(directory) / "nodes.json"
                    nodes.write_text(json.dumps([{"metadata": {"name": "spark-38fc", "labels": {
                        "nvidia.com/gpu.product": "NVIDIA-GB10"}}, "spec": {"providerID": ""}}]))
                    decoded = json.loads(subprocess.check_output([
                        os.environ["NVCRECTL"], "workflow", "render", str(path),
                        "--nodes-file", str(nodes), "--output", "json"], text=True))
                    self.assert_contract(decoded, rail)

        diagnose = json.loads(subprocess.check_output([
            os.environ["NVCRECTL"], "certification", "render", "--platform", "onprem", "--output", "json",
            str(Path(__file__).with_name("certification-diagnose.json"))], text=True))
        self.assertEqual(len(diagnose), 1)
        diagnose_spec = diagnose[0]["spec"]["orchestration"]["diagnose"]
        self.assertEqual(diagnose_spec["minGroupSize"], 2)
        self.assertNotIn("topologyKey", diagnose_spec)
        self.assertNotIn("NCCL_MNNVL_ENABLE=1",
                         diagnose[0]["spec"]["jobTemplate"]["spec"]["workload"]["trainJob"]["trainer"]["args"])


if __name__ == "__main__":
    unittest.main()

---
title: C2CMeasurement
description: CRD reference for CPU-GPU coherent-memory measurements.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

`C2CMeasurement` watches an NVCRE `Job` and parses the stable result records
emitted by the GB10 C2C benchmark. It records CPU-to-GPU and GPU-to-CPU
bandwidth and latency for managed and mapped pinned memory, together with the
benchmark's correctness result.

The Job controller creates this resource automatically when
`spec.c2cMeasurement` is present. The `diagnostics/gb10-c2c` catalog category
uses that path and runs one Job per selected node.

## Spec

```yaml
spec:
  jobRef:
    apiGroup: nvcre.nvidia.com
    kind: Job
    name: gb10-c2c-job
  sampleInterval: 1s
```

## Status

Each `status.results` row contains:

- `direction`: `cpuToGPU` or `gpuToCPU`
- `memoryType`: `managed` or `pinned`
- `sizeBytes`: transfer size
- `bandwidthGBps`: average decimal GB/s
- `latencyUs`: average operation latency
- `samples`: benchmark iterations represented by the row
- `verified`: whether cross-processor data validation passed

The associated Job can enforce `c2cCPUToGPUBandwidthGBps` and
`c2cGPUToCPUBandwidthGBps` CEL thresholds. These use the slowest allocation
path at the largest measured size. `c2cVerified` is 1 only when every result is
verified. Correctness failures also make the benchmark exit non-zero,
independently of performance thresholds.

Prometheus exports `nvcre_c2c_bandwidth_gbps`,
`nvcre_c2c_latency_microseconds`, and `nvcre_c2c_verified` while the
measurement is active. `nvcrectl certification report` includes the retained
status rows after completion.

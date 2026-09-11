// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package c2c parses the stable output emitted by the GB10 C2C benchmark.
package c2c

import (
	"regexp"
	"strconv"
)

var resultPattern = regexp.MustCompile(`NVCRE_C2C_RESULT direction=(cpuToGPU|gpuToCPU) memory=([A-Za-z0-9_-]+) size_bytes=([0-9]+) bandwidth_gbps=([0-9]+(?:\.[0-9]+)?) latency_us=([0-9]+(?:\.[0-9]+)?) samples=([0-9]+) verified=(true|false)`)

// DataPoint is one parsed C2C benchmark summary.
type DataPoint struct {
	Direction     string
	MemoryType    string
	SizeBytes     int64
	BandwidthGBps float64
	LatencyUs     float64
	Samples       int
	Verified      bool
}

// Parse extracts valid benchmark rows and ignores unrelated log lines.
func Parse(lines []string) []DataPoint {
	results := make([]DataPoint, 0)
	for _, line := range lines {
		m := resultPattern.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		size, err1 := strconv.ParseInt(m[3], 10, 64)
		bw, err2 := strconv.ParseFloat(m[4], 64)
		latency, err3 := strconv.ParseFloat(m[5], 64)
		samples, err4 := strconv.Atoi(m[6])
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil || size <= 0 || samples <= 0 {
			continue
		}
		results = append(results, DataPoint{
			Direction: m[1], MemoryType: m[2], SizeBytes: size,
			BandwidthGBps: bw, LatencyUs: latency, Samples: samples,
			Verified: m[7] == "true",
		})
	}
	return results
}

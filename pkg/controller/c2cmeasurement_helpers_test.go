// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	nvcrev1alpha1 "github.com/NVIDIA/cluster-readiness-engine/api/v1alpha1"
	"github.com/NVIDIA/cluster-readiness-engine/pkg/c2c"
)

const testC2CMemoryManaged = "managed"

func TestMergeC2CResultsWeightedAndStickyFailure(t *testing.T) {
	existing := []nvcrev1alpha1.C2CResult{{
		Direction: nvcrev1alpha1.C2CDirectionCPUToGPU, MemoryType: testC2CMemoryManaged, SizeBytes: 1024,
		BandwidthGBps: "10.00", LatencyUs: "2.00", Samples: 2, Verified: true,
	}}
	got := mergeC2CResults(existing, []c2c.DataPoint{{
		Direction: nvcrev1alpha1.C2CDirectionCPUToGPU, MemoryType: testC2CMemoryManaged, SizeBytes: 1024,
		BandwidthGBps: 20, LatencyUs: 4, Samples: 2, Verified: false,
	}})
	if len(got) != 1 || got[0].BandwidthGBps != "15.00" || got[0].LatencyUs != "3.00" || got[0].Samples != 4 {
		t.Fatalf("unexpected weighted result: %#v", got)
	}
	if got[0].Verified {
		t.Fatal("a correctness failure must remain sticky")
	}
	if !existing[0].Verified || existing[0].Samples != 2 {
		t.Fatalf("input mutated: %#v", existing)
	}
}

func TestC2CDirectionBandwidthUsesSlowestPathAtLargestSize(t *testing.T) {
	results := []nvcrev1alpha1.C2CResult{
		{Direction: nvcrev1alpha1.C2CDirectionCPUToGPU, MemoryType: testC2CMemoryManaged, SizeBytes: 1024, BandwidthGBps: "99"},
		{Direction: nvcrev1alpha1.C2CDirectionCPUToGPU, MemoryType: testC2CMemoryManaged, SizeBytes: 2048, BandwidthGBps: "20"},
		{Direction: nvcrev1alpha1.C2CDirectionCPUToGPU, MemoryType: "pinned", SizeBytes: 2048, BandwidthGBps: "12"},
	}
	got, ok := c2cDirectionBandwidth(results, nvcrev1alpha1.C2CDirectionCPUToGPU)
	if !ok || got != 12 {
		t.Fatalf("got (%v, %v), want (12, true)", got, ok)
	}
}

func TestMergeC2CResultsIgnoresRepeatedLogRows(t *testing.T) {
	point := c2c.DataPoint{
		Direction: nvcrev1alpha1.C2CDirectionCPUToGPU, MemoryType: testC2CMemoryManaged,
		SizeBytes: 1024, BandwidthGBps: 20, LatencyUs: 4, Samples: 5, Verified: true,
	}
	results := mergeC2CResults(nil, []c2c.DataPoint{point})
	results = mergeC2CResults(results, []c2c.DataPoint{point})
	if len(results) != 1 || results[0].Samples != 5 {
		t.Fatalf("repeated rows changed the result: %#v", results)
	}
}

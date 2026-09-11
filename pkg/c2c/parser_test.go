// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package c2c

import "testing"

func TestParse(t *testing.T) {
	lines := []string{
		"noise",
		"NVCRE_C2C_RESULT direction=cpuToGPU memory=managed size_bytes=67108864 bandwidth_gbps=91.25 latency_us=735.42 samples=20 verified=true",
		"NVCRE_C2C_RESULT direction=gpuToCPU memory=system size_bytes=4096 bandwidth_gbps=2.50 latency_us=1.64 samples=5 verified=false",
	}
	got := Parse(lines)
	if len(got) != 2 || got[0].Direction != "cpuToGPU" || got[0].SizeBytes != 67108864 || !got[0].Verified {
		t.Fatalf("unexpected parse result: %#v", got)
	}
	if got[1].MemoryType != "system" || got[1].Verified {
		t.Fatalf("unexpected second result: %#v", got[1])
	}
}

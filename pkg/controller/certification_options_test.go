// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	"github.com/stretchr/testify/assert"

	nvcrev1alpha1 "github.com/NVIDIA/cluster-readiness-engine/api/v1alpha1"
)

func TestResolveOptionsNICResources(t *testing.T) {
	global := nvcrev1alpha1.CategoryOptions{NicResources: []nvcrev1alpha1.NICResource{
		{Name: "rdma/global", Quantity: 2},
	}}
	override := nvcrev1alpha1.CategoryOptions{NicResources: []nvcrev1alpha1.NICResource{
		{Name: "rdma/rail_a", Quantity: 1},
		{Name: "rdma/rail_b", Quantity: 1},
	}}

	resolved := ResolveOptions(&global, &override)

	assert.Equal(t, override.NicResources, resolved.NicResources)
}

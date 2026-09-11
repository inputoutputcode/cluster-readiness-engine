// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	C2CMeasurementMeasuring = "Measuring"
	C2CMeasurementComplete  = "Complete"
	C2CDirectionCPUToGPU    = "cpuToGPU"
	C2CDirectionGPUToCPU    = "gpuToCPU"
)

// C2CMeasurementSpec defines a CPU-GPU coherent-memory measurement.
type C2CMeasurementSpec struct {
	// jobRef is the NVCRE Job whose worker logs contain C2C results.
	// +optional
	JobRef corev1.TypedLocalObjectReference `json:"jobRef,omitempty"`

	// sampleInterval is how often worker logs are sampled. Default: 60s.
	// +optional
	SampleInterval *metav1.Duration `json:"sampleInterval,omitempty"`
}

// C2CResult stores an averaged CPU-GPU transfer result.
type C2CResult struct {
	// direction is cpuToGPU or gpuToCPU.
	// +kubebuilder:validation:Enum=cpuToGPU;gpuToCPU
	Direction string `json:"direction"`

	// memoryType identifies the allocation path, such as managed or system.
	MemoryType string `json:"memoryType"`

	// sizeBytes is the measured transfer size.
	SizeBytes int64 `json:"sizeBytes"`

	// bandwidthGBps is the average decimal GB/s throughput.
	BandwidthGBps string `json:"bandwidthGBps"`

	// latencyUs is the average operation latency in microseconds.
	LatencyUs string `json:"latencyUs"`

	// samples is the number of benchmark iterations represented by this row.
	Samples int `json:"samples"`

	// verified reports whether the benchmark's cross-processor correctness check passed.
	Verified bool `json:"verified"`
}

// C2CMeasurementStatus defines the observed state of C2CMeasurement.
type C2CMeasurementStatus struct {
	// results contains per-direction, per-memory-type, per-size measurements.
	// +optional
	Results []C2CResult `json:"results,omitempty"`

	// startTime is when the first C2C result was collected.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// completionTime is when the referenced Job became terminal.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// conditions represent measurement progress and completion.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// C2CMeasurement is the Schema for CPU-GPU coherent-memory measurements.
type C2CMeasurement struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired measurement.
	// +required
	Spec C2CMeasurementSpec `json:"spec"`

	// status contains observed C2C results.
	// +optional
	Status C2CMeasurementStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// C2CMeasurementList contains a list of C2CMeasurement resources.
type C2CMeasurementList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []C2CMeasurement `json:"items"`
}

func init() {
	Register(&C2CMeasurement{}, &C2CMeasurementList{})
}

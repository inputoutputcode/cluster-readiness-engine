// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"

	meta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvcrev1alpha1 "github.com/NVIDIA/cluster-readiness-engine/api/v1alpha1"
)

// missingJobThresholdKeys returns threshold keys that have no corresponding measured value.
func missingJobThresholdKeys(thresholds map[string]string, measured map[string]float64) []string {
	var missing []string
	for key := range thresholds {
		if _, ok := measured[key]; !ok {
			missing = append(missing, key)
		}
	}
	return missing
}

// findJobGoodputMeasurement returns the first GoodputMeasurement in the namespace
// that references this Job, or nil if none exists.
func findJobGoodputMeasurement(ctx context.Context, c client.Reader, job *nvcrev1alpha1.Job) *nvcrev1alpha1.GoodputMeasurement {
	var measurements nvcrev1alpha1.GoodputMeasurementList
	if err := c.List(ctx, &measurements, matchingJobRef(job.Namespace, job.Name)...); err != nil {
		return nil
	}
	if len(measurements.Items) == 0 {
		return nil
	}
	return &measurements.Items[0]
}

// findJobBandwidthMeasurement returns the first BandwidthMeasurement in the namespace
// that references this Job, or nil if none exists.
func findJobBandwidthMeasurement(ctx context.Context, c client.Reader, job *nvcrev1alpha1.Job) *nvcrev1alpha1.BandwidthMeasurement {
	var measurements nvcrev1alpha1.BandwidthMeasurementList
	if err := c.List(ctx, &measurements, matchingJobRef(job.Namespace, job.Name)...); err != nil {
		return nil
	}
	if len(measurements.Items) == 0 {
		return nil
	}
	return &measurements.Items[0]
}

func findJobC2CMeasurement(ctx context.Context, c client.Reader, job *nvcrev1alpha1.Job) *nvcrev1alpha1.C2CMeasurement {
	var measurements nvcrev1alpha1.C2CMeasurementList
	if err := c.List(ctx, &measurements, matchingJobRef(job.Namespace, job.Name)...); err != nil || len(measurements.Items) == 0 {
		return nil
	}
	return &measurements.Items[0]
}

// collectJobMeasuredValues gathers metric values from BandwidthMeasurement and
// GoodputMeasurement status fields. Keys match the threshold registry.
func collectJobMeasuredValues(ctx context.Context, c client.Reader, job *nvcrev1alpha1.Job) map[string]float64 {
	values := make(map[string]float64)

	if bm := findJobBandwidthMeasurement(ctx, c, job); bm != nil && len(bm.Status.Results) > 0 {
		values["busBandwidthGBps"] = maxBusBandwidth(bm.Status.Results)
		values["algBandwidthGBps"] = maxAlgBandwidth(bm.Status.Results)
	}

	if cm := findJobC2CMeasurement(ctx, c, job); cm != nil &&
		meta.IsStatusConditionTrue(cm.Status.Conditions, nvcrev1alpha1.C2CMeasurementComplete) {
		if value, ok := c2cDirectionBandwidth(cm.Status.Results, nvcrev1alpha1.C2CDirectionCPUToGPU); ok {
			values["c2cCPUToGPUBandwidthGBps"] = value
		}
		if value, ok := c2cDirectionBandwidth(cm.Status.Results, nvcrev1alpha1.C2CDirectionGPUToCPU); ok {
			values["c2cGPUToCPUBandwidthGBps"] = value
		}
		values["c2cVerified"] = 1
		for _, result := range cm.Status.Results {
			if !result.Verified {
				values["c2cVerified"] = 0
				break
			}
		}
	}

	// Goodput-derived values are provisional until the measurement's Complete
	// condition is True: the GoodputMeasurement controller freezes the status
	// in a single terminal write anchored to the Job's terminal transition
	// (ADR-072). Evaluating earlier would make pass/fail depend on when this
	// controller happened to read. The missing-key requeue and the
	// measurementTimeout machinery in checkPerformanceThresholds handle the
	// wait for Complete.
	if gm := findJobGoodputMeasurement(ctx, c, job); gm != nil &&
		meta.IsStatusConditionTrue(gm.Status.Conditions, nvcrev1alpha1.GoodputMeasurementComplete) {
		if v := parseStallFloat(gm.Status.Result); v > 0 {
			values["goodputRatio"] = v
		}
		if v := parseStallFloat(gm.Status.AvgTFLOPSPerGPU); v > 0 {
			values["avgTFLOPsPerGPU"] = v
		}
		if v := parseStallFloat(gm.Status.AvgStepTimeSec); v > 0 {
			values["avgStepTimeSec"] = v
		}
	}
	return values
}

// c2cDirectionBandwidth returns the slowest allocation path at the largest
// measured size. Every memory type must clear the threshold; a fast path must
// not hide a broken or unexpectedly slow path.
func c2cDirectionBandwidth(results []nvcrev1alpha1.C2CResult, direction string) (float64, bool) {
	var largest int64
	for _, result := range results {
		if result.Direction == direction && result.SizeBytes > largest {
			largest = result.SizeBytes
		}
	}
	var value float64
	found := false
	for _, result := range results {
		if result.Direction != direction || result.SizeBytes != largest {
			continue
		}
		bandwidth := parseStallFloat(result.BandwidthGBps)
		if !found || bandwidth < value {
			value, found = bandwidth, true
		}
	}
	return value, found
}

// isJobAwaitingThresholdEvaluation returns true when a succeeded Job has performance
// thresholds configured but the Job controller has not yet recorded the outcome.
// The Job controller sets ValidationFailed=True on violation or ValidationFailed=False
// when all thresholds pass. Workflow should keep groups running until one is set.
func isJobAwaitingThresholdEvaluation(job *nvcrev1alpha1.Job) bool {
	if len(job.Spec.Thresholds) == 0 {
		return false
	}
	succeededCond := meta.FindStatusCondition(job.Status.Conditions, nvcrev1alpha1.JobSucceeded)
	if succeededCond == nil || succeededCond.Status != metav1.ConditionTrue {
		return false
	}
	return meta.FindStatusCondition(job.Status.Conditions, nvcrev1alpha1.JobValidationFailed) == nil
}

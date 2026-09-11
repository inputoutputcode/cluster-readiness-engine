// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	controlleropts "sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	nvcrev1alpha1 "github.com/NVIDIA/cluster-readiness-engine/api/v1alpha1"
	"github.com/NVIDIA/cluster-readiness-engine/pkg/c2c"
	"github.com/NVIDIA/cluster-readiness-engine/pkg/podlogs"
	"github.com/NVIDIA/cluster-readiness-engine/pkg/podutil"
)

const (
	defaultC2CMeasurementRequeueInterval = 15 * time.Second
	c2cMeasurementFinalizer              = "nvcre.nvidia.com/c2cmeasurement-finalizer"
	reasonC2CJobRunning                  = "JobRunning"
	reasonC2CJobSucceeded                = "JobSucceeded"
	reasonC2CJobFailed                   = "JobFailed"
	reasonC2CNoData                      = "NoDataCollected"
)

// C2CMeasurementReconciler collects CPU-GPU coherent-memory benchmark output.
type C2CMeasurementReconciler struct {
	client.Client
	APIReader               client.Reader
	Scheme                  *runtime.Scheme
	Clientset               *kubernetes.Clientset
	LogFetcher              podlogs.PodLogFetcher
	MaxConcurrentReconciles int
	RequeueInterval         time.Duration

	mu         sync.Mutex
	lastSample map[string]time.Time
}

func (r *C2CMeasurementReconciler) podReader() client.Reader {
	if r.APIReader != nil {
		return r.APIReader
	}
	return r.Client
}

func (r *C2CMeasurementReconciler) fetcher() podlogs.PodLogFetcher {
	if r.LogFetcher != nil {
		return r.LogFetcher
	}
	return podlogs.NewKubernetesLogFetcher(r.Clientset)
}

func (r *C2CMeasurementReconciler) sampleInterval(m *nvcrev1alpha1.C2CMeasurement) time.Duration {
	if m.Spec.SampleInterval != nil {
		return m.Spec.SampleInterval.Duration
	}
	return defaultSampleInterval
}

func (r *C2CMeasurementReconciler) requeueInterval() time.Duration {
	if r.RequeueInterval > 0 {
		return r.RequeueInterval
	}
	return defaultC2CMeasurementRequeueInterval
}

// +kubebuilder:rbac:groups=nvcre.nvidia.com,resources=c2cmeasurements,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvcre.nvidia.com,resources=c2cmeasurements/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvcre.nvidia.com,resources=c2cmeasurements/finalizers,verbs=update

func (r *C2CMeasurementReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	m := &nvcrev1alpha1.C2CMeasurement{}
	if err := r.Get(ctx, req.NamespacedName, m); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !m.DeletionTimestamp.IsZero() {
		return r.handleC2CDeletion(ctx, m)
	}
	if added, err := ensureFinalizer(ctx, r.Client, m, c2cMeasurementFinalizer); err != nil || added {
		return ctrl.Result{}, err
	}
	if meta.IsStatusConditionTrue(m.Status.Conditions, nvcrev1alpha1.C2CMeasurementComplete) {
		return ctrl.Result{}, nil
	}
	if m.Spec.JobRef.Name == "" {
		return ctrl.Result{}, nil
	}

	job := &nvcrev1alpha1.Job{}
	key := types.NamespacedName{Name: m.Spec.JobRef.Name, Namespace: m.Namespace}
	if err := r.Get(ctx, key, job); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{RequeueAfter: r.requeueInterval()}, nil
		}
		return ctrl.Result{}, fmt.Errorf("get referenced Job: %w", err)
	}

	if meta.IsStatusConditionTrue(job.Status.Conditions, nvcrev1alpha1.JobSucceeded) {
		if len(m.Status.Results) == 0 {
			_ = r.sampleC2C(ctx, m, job, true)
		}
		return r.completeC2C(ctx, m, reasonC2CJobSucceeded, "Job succeeded")
	}
	if meta.IsStatusConditionTrue(job.Status.Conditions, nvcrev1alpha1.JobFailed) {
		if len(m.Status.Results) == 0 {
			_ = r.sampleC2C(ctx, m, job, true)
		}
		return r.completeC2C(ctx, m, reasonC2CJobFailed, "Job failed")
	}
	if meta.IsStatusConditionTrue(job.Status.Conditions, nvcrev1alpha1.JobInProgress) {
		if err := r.sampleC2C(ctx, m, job, false); err != nil {
			logf.FromContext(ctx).Info("C2C result not available yet", "error", err)
		}
		return ctrl.Result{RequeueAfter: r.sampleInterval(m)}, nil
	}
	return ctrl.Result{RequeueAfter: r.requeueInterval()}, nil
}

func (r *C2CMeasurementReconciler) sampleC2C(
	ctx context.Context, m *nvcrev1alpha1.C2CMeasurement, job *nvcrev1alpha1.Job, force bool,
) error {
	key := m.Namespace + "/" + m.Name
	r.mu.Lock()
	last := r.lastSample[key]
	r.mu.Unlock()
	if !force && !last.IsZero() && time.Since(last) < r.sampleInterval(m) {
		return nil
	}
	if job.Status.WorkloadRef == nil || job.Status.WorkloadRef.Name == "" {
		return fmt.Errorf("job has no workload reference")
	}
	pod, err := podutil.NewWorkerDiscoverer(r.podReader()).GetReplicatedJobPod(
		ctx, m.Namespace, job.Status.WorkloadRef.Name, labelNode)
	if err != nil {
		return fmt.Errorf("find C2C worker pod: %w", err)
	}
	if !podutil.IsPodRunning(pod) && pod.Status.Phase != corev1.PodSucceeded && pod.Status.Phase != corev1.PodFailed {
		return fmt.Errorf("C2C worker pod is %s", pod.Status.Phase)
	}
	lines, err := r.fetcher().FetchLogs(ctx, m.Namespace, pod.Name, podlogs.LogOptions{Container: labelNode})
	if err != nil {
		return fmt.Errorf("fetch C2C logs: %w", err)
	}
	points := c2c.Parse(lines)
	if len(points) == 0 {
		return fmt.Errorf("no C2C result rows parsed")
	}
	m.Status.Results = mergeC2CResults(m.Status.Results, points)
	if m.Status.StartTime == nil {
		now := metav1.Now()
		m.Status.StartTime = &now
	}
	meta.SetStatusCondition(&m.Status.Conditions, metav1.Condition{
		Type: nvcrev1alpha1.C2CMeasurementMeasuring, Status: metav1.ConditionTrue,
		ObservedGeneration: m.Generation, Reason: reasonC2CJobRunning,
		Message: measurementInProgressMessage,
	})
	if err := r.Status().Update(ctx, m); err != nil {
		return fmt.Errorf("update C2CMeasurement status: %w", err)
	}
	recordC2CMetrics(m.Namespace, m.Name, job.Name, m.Labels["nvcre.nvidia.com/workflow"], m.Status.Results)
	r.mu.Lock()
	if r.lastSample == nil {
		r.lastSample = make(map[string]time.Time)
	}
	r.lastSample[key] = time.Now()
	r.mu.Unlock()
	return nil
}

func mergeC2CResults(existing []nvcrev1alpha1.C2CResult, points []c2c.DataPoint) []nvcrev1alpha1.C2CResult {
	results := append([]nvcrev1alpha1.C2CResult(nil), existing...)
	index := make(map[string]int, len(results))
	for i := range results {
		index[results[i].Direction+"/"+results[i].MemoryType+"/"+strconv.FormatInt(results[i].SizeBytes, 10)] = i
	}
	for _, p := range points {
		key := p.Direction + "/" + p.MemoryType + "/" + strconv.FormatInt(p.SizeBytes, 10)
		if i, ok := index[key]; ok {
			oldBW, _ := strconv.ParseFloat(results[i].BandwidthGBps, 64)
			oldLatency, _ := strconv.ParseFloat(results[i].LatencyUs, 64)
			if oldBW == p.BandwidthGBps && oldLatency == p.LatencyUs &&
				results[i].Samples == p.Samples && results[i].Verified == p.Verified {
				continue
			}
			oldN := results[i].Samples
			newN := oldN + p.Samples
			results[i].BandwidthGBps = strconv.FormatFloat((oldBW*float64(oldN)+p.BandwidthGBps*float64(p.Samples))/float64(newN), 'f', 2, 64)
			results[i].LatencyUs = strconv.FormatFloat((oldLatency*float64(oldN)+p.LatencyUs*float64(p.Samples))/float64(newN), 'f', 2, 64)
			results[i].Samples = newN
			results[i].Verified = results[i].Verified && p.Verified
			continue
		}
		index[key] = len(results)
		results = append(results, nvcrev1alpha1.C2CResult{
			Direction: p.Direction, MemoryType: p.MemoryType, SizeBytes: p.SizeBytes,
			BandwidthGBps: strconv.FormatFloat(p.BandwidthGBps, 'f', 2, 64),
			LatencyUs:     strconv.FormatFloat(p.LatencyUs, 'f', 2, 64), Samples: p.Samples, Verified: p.Verified,
		})
	}
	return results
}

func (r *C2CMeasurementReconciler) completeC2C(
	ctx context.Context, m *nvcrev1alpha1.C2CMeasurement, reason, message string,
) (ctrl.Result, error) {
	if reason == reasonC2CJobSucceeded && len(m.Status.Results) == 0 {
		reason, message = reasonC2CNoData, "Job succeeded but no C2C data was parsed"
	}
	now := metav1.Now()
	m.Status.CompletionTime = &now
	meta.SetStatusCondition(&m.Status.Conditions, metav1.Condition{Type: nvcrev1alpha1.C2CMeasurementMeasuring, Status: metav1.ConditionFalse, ObservedGeneration: m.Generation, Reason: reason, Message: message})
	meta.SetStatusCondition(&m.Status.Conditions, metav1.Condition{Type: nvcrev1alpha1.C2CMeasurementComplete, Status: metav1.ConditionTrue, ObservedGeneration: m.Generation, Reason: reason, Message: message})
	if err := r.Status().Update(ctx, m); err != nil {
		return ctrl.Result{}, fmt.Errorf("complete C2CMeasurement: %w", err)
	}
	cleanupC2CMetrics(m.Namespace, m.Name, m.Spec.JobRef.Name)
	return ctrl.Result{}, nil
}

func (r *C2CMeasurementReconciler) handleC2CDeletion(ctx context.Context, m *nvcrev1alpha1.C2CMeasurement) (ctrl.Result, error) {
	if controllerutil.ContainsFinalizer(m, c2cMeasurementFinalizer) {
		cleanupC2CMetrics(m.Namespace, m.Name, m.Spec.JobRef.Name)
		r.mu.Lock()
		delete(r.lastSample, m.Namespace+"/"+m.Name)
		r.mu.Unlock()
		controllerutil.RemoveFinalizer(m, c2cMeasurementFinalizer)
		if err := r.Update(ctx, m); err != nil {
			return ctrl.Result{}, fmt.Errorf("remove C2CMeasurement finalizer: %w", err)
		}
	}
	return ctrl.Result{}, nil
}

func (r *C2CMeasurementReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvcrev1alpha1.C2CMeasurement{}).
		Named("c2cmeasurement").
		WithOptions(controlleropts.Options{MaxConcurrentReconciles: r.MaxConcurrentReconciles}).
		Complete(r)
}

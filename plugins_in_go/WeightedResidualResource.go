package weightedresidualresource

import (
	"context"
	"log"
	"k8s.io/apimachinery/pkg/runtime" // Kept the Kubernetes runtime import
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	Name  = "WeightedResidualResource"
	Alpha = 0.5
	Beta  = 0.5
	Gamma = 0.5
)

type WeightedResidualResource struct {
	handle framework.Handle
}

var _ framework.ScorePlugin = &WeightedResidualResource{}

func (pl *WeightedResidualResource) Name() string {
	return Name
}

func (pl *WeightedResidualResource) Score(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		log.Printf("Failed to get node info for %s: %v", nodeName, err)
		return 0, framework.AsStatus(err)
	}

	// Extract CPU and memory requests and limits from the pod spec
	// This example assumes the first container in the pod's spec
	container := pod.Spec.Containers[0]
	cpuRequest := container.Resources.Requests.Cpu().MilliValue()
	memoryRequest := container.Resources.Requests.Memory().Value()
	cpuLimit := container.Resources.Limits.Cpu().MilliValue()
	memoryLimit := container.Resources.Limits.Memory().Value()

	// Apply the weighted residual resource scoring method
	score := weightedResidualResource(nodeInfo, cpuRequest, memoryRequest, cpuLimit, memoryLimit, Alpha, Beta, Gamma)

	return score, nil
}

func weightedResidualResource(nodeInfo *framework.NodeInfo, cpuRequest, memoryRequest, cpuLimit, memoryLimit int64, alpha, beta, gamma float64) int64 {
	availableCpu := nodeInfo.Allocatable.MilliCPU - nodeInfo.Requested.MilliCPU
	availableMemory := nodeInfo.Allocatable.Memory - nodeInfo.Requested.Memory

	// Adjust CPU and memory scores based on the limits and available resources
	cpuScore := float64(cpuRequest)
	if float64(cpuLimit) > float64(beta)*float64(availableCpu) {
		cpuScore = float64(cpuLimit)
	}
	memoryScore := float64(memoryRequest)
	if float64(memoryLimit) > float64(gamma)*float64(availableMemory) {
		memoryScore = float64(memoryLimit)
	}

	// Combine the scores using the alpha weight and return the final weighted score
	weightedScore := alpha*cpuScore + (1-alpha)*memoryScore
	return int64(weightedScore)
}

func (pl *WeightedResidualResource) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// Updated NewWeightedResidualResourcePlugin function to include context.Context parameter
func NewWeightedResidualResourcePlugin(ctx context.Context, configuration runtime.Object, f framework.Handle) (framework.Plugin, error) {
	return &WeightedResidualResource{handle: f}, nil
}

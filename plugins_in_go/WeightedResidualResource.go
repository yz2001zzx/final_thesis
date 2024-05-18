package weightedresidualresource

import (
	"context" // Provides functionalities to carry deadlines, cancellation signals, and other request-scoped values across API boundaries.
	"log"     // Implements simple logging.
	"math"    // Provides basic constants and mathematical functions.

	"k8s.io/apimachinery/pkg/runtime"           // Provides utilities for working with API objects.
	"k8s.io/kubernetes/pkg/scheduler/framework" // Import the Kubernetes scheduler framework to implement custom scheduling plugins.
	"k8s.io/api/core/v1"// Import the Kubernetes core API to access Pod and Node types.
)

// Declare constants for the plugin name and the weights used in scoring.
const (
	Name  = "WeightedResidualResource" // Name of the plugin
	Alpha = 0.5                        // Weight for CPU in the scoring function
	Beta  = 0.5                        // Weight for CPU limit in the scoring function
	Gamma = 0.5                        // Weight for Memory limit in the scoring function
)

// WeightedResidualResourcePlugin is a struct that holds the handle to interact with the Kubernetes scheduler framework.
type WeightedResidualResourcePlugin struct {
	handle framework.Handle // The handle provides methods to interact with the scheduler framework.
}

// Compile-time assertion to ensure WeightedResidualResourcePlugin implements the ScorePlugin interface.
// This ensures that WeightedResidualResourcePlugin conforms to the required interface.
var _ framework.ScorePlugin = &WeightedResidualResourcePlugin{}

// Name method returns the name of the plugin. It's required by the ScorePlugin interface.
func (pl *WeightedResidualResourcePlugin) Name() string {
	return Name // Returns the name of this plugin.
}

// Score method is invoked at the score extension point. It's where the scoring logic is executed.
// This method calculates a score for a given node based on the pod's resource requests.
func (pl *WeightedResidualResourcePlugin) Score(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	// Retrieve node information using the scheduler framework.
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		// Log the error and return a score of 0 if node information cannot be retrieved.
		log.Printf("Failed to get node info for %s: %v", nodeName, err)
		return 0, framework.AsStatus(err)
	}

	// Extract CPU and memory requests and limits from the pod spec.
	// This example assumes the first container in the pod's spec.
	// In the PoC, we do not consider multi-container use cases.
	cpuRequest := pod.Spec.Containers[0].Resources.Requests.Cpu().MilliValue()  // CPU request in milli-units.
	memoryRequest := pod.Spec.Containers[0].Resources.Requests.Memory().Value() // Memory request in bytes.
	cpuLimit := pod.Spec.Containers[0].Resources.Limits.Cpu().MilliValue()      // CPU limit in milli-units.
	memoryLimit := pod.Spec.Containers[0].Resources.Limits.Memory().Value()     // Memory limit in bytes.

	// Apply the weighted residual resource scoring method.
	score := weightedResidualResource(nodeInfo, cpuRequest, memoryRequest, cpuLimit, memoryLimit, Alpha, Beta, Gamma)

	// Return the calculated score and a nil status indicating success.
	return score, nil
}

// weightedResidualResource function calculates the score based on available CPU and memory resources,
// and pod resource requests and limits.
func weightedResidualResource(nodeInfo *framework.NodeInfo, cpuRequest, memoryRequest, cpuLimit, memoryLimit int64, alpha, beta, gamma float64) int64 {
	// Calculate available CPU and memory on the node.
	availableCpu := nodeInfo.Node().Status.Allocatable.Cpu().MilliValue() - nodeInfo.Requested.Cpu().MilliValue()
	availableMemory := nodeInfo.Node().Status.Allocatable.Memory().Value() - nodeInfo.Requested.Memory().Value()

	// Adjust CPU and memory scores based on the limits and available resources.
	// The max and min functions are used to ensure scores are within the range of the request and limit.
	cpuScore := math.Max(math.Min(beta*float64(availableCpu), float64(cpuLimit)), float64(cpuRequest))
	memoryScore := math.Max(math.Min(gamma*float64(availableMemory), float64(memoryLimit)), float64(memoryRequest))

	// Combine the scores using the alpha weight and return the final weighted score.
	weightedScore := alpha*cpuScore + (1-alpha)*memoryScore
	return int64(weightedScore)
}

// ScoreExtensions method returns nil because this plugin does not implement any score extensions.
// This method is part of the ScorePlugin interface and must be defined, even if it just returns nil.
func (pl *WeightedResidualResourcePlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// Added the factory function.
// NewWeightedResidualResourcePlugin is the factory function to create an instance of WeightedResidualResourcePlugin.
// It parses the configuration and initializes the plugin.
func NewWeightedResidualResourcePlugin(configuration runtime.Object, f framework.Handle) (framework.Plugin, error) {
	// Create a new instance of WeightedResidualResourcePlugin with the provided handle.
	return &WeightedResidualResourcePlugin{handle: f}, nil
}

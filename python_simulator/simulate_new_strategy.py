import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

class Node:
    def __init__(self, cpu, memory):
        self.total_cpu = cpu
        self.total_memory = memory
        self.used_cpu = 0
        self.used_memory = 0

    def allocate_resources(self, cpu, memory):
        self.used_cpu += cpu
        self.used_memory += memory

    def reset_resources(self):
        self.used_cpu = 0
        self.used_memory = 0

    def utilization_percentage(self):
        cpu_util = self.used_cpu / self.total_cpu * 100
        memory_util = self.used_memory / self.total_memory * 100
        return cpu_util, memory_util

    def __repr__(self):
        return f"Node(CPU Available: {self.total_cpu - self.used_cpu}/{self.total_cpu}, Memory Available: {self.total_memory - self.used_memory}/{self.total_memory})"

def simulate_scheduling(scoring_function, strategy_name, alpha=0.25, beta=0.96, gamma=0.69):
    util_data = []
    cpu_stds = []
    memory_stds = []
    failed_pods = 0
    
    print(f"\nSimulating {strategy_name} Strategy...")
    random.seed(42)  # for reproducibility
    for i in range(500):
        cpu_request = random.randint(1, 4)
        memory_request = random.randint(1, 16)
        cpu_limit = random.randint(cpu_request, cpu_request + 2)
        memory_limit = random.randint(memory_request, memory_request + 32)
        
        scored_nodes = [(node, scoring_function(node, cpu_request, memory_request, cpu_limit, memory_limit, alpha, beta, gamma), idx) for idx, node in enumerate(nodes)]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        print(f"\nRound {i + 1} - Pod Request (CPU: {cpu_request}, Memory: {memory_request}), Limits (CPU: {cpu_limit}, Memory: {memory_limit})")
        print("Ranking of Nodes Before Allocation:")
        for rank, (node, score, idx) in enumerate(scored_nodes):
            print(f"Rank {rank + 1} - Node {idx}: {str(node)}, Score: {score:.2f}")

        allocated = False # Flag to check if scheduling happens
        for node, score, idx in scored_nodes:
            if node.total_cpu - node.used_cpu >= cpu_request and node.total_memory - node.used_memory >= memory_request:
                actual_cpu = random.randint(cpu_request, cpu_limit)
                actual_memory = random.randint(memory_request, memory_limit)
                if node.total_cpu - node.used_cpu >= actual_cpu and node.total_memory - node.used_memory >= actual_memory:
                    print(f"Selecting Node {idx} for Allocation:")
                    print(f"Before Allocation - Node {idx}: {str(node)}")
                    node.allocate_resources(actual_cpu, actual_memory)
                    print(f"After Allocation - Node {idx}: {str(node)}")
                    allocated = True
                    break
        if not allocated:
            print("The pod cannot be scheduled in this round")
            failed_pods += 1  # Increment the counter if no node could schedule the pod
                

        cpu_utilizations = [node.utilization_percentage()[0] for node in nodes]
        memory_utilizations = [node.utilization_percentage()[1] for node in nodes]
        ave_cpu = sum(cpu_utilizations) / len(nodes)
        ave_memory = sum(memory_utilizations) / len(nodes)
        cpu_std = np.std(cpu_utilizations)
        memory_std = np.std(memory_utilizations)
        cpu_stds.append(cpu_std)
        memory_stds.append(memory_std)
        print(
            f"After the Round {i + 1}: Average CPU Utilization: {ave_cpu}%, "
            f"Average Memory Utilization: {ave_memory}%, CPU Standard Deviation: {cpu_std}, "
            f"Memory Standard Deviation: {memory_std}"
        )
        util_data.append([i + 1, ave_cpu, ave_memory])

    for node in nodes:
        node.reset_resources()

    df = pd.DataFrame(util_data, columns=['Round', 'Average CPU Utilization', 'Average Memory Utilization'])
    std_df = pd.DataFrame({
        'Round': range(1, 501),
        'CPU Utilization Std Dev': cpu_stds,
        'Memory Utilization Std Dev': memory_stds
    })
    df.to_csv(f"{strategy_name}_utilization.csv", index=False)
    std_df.to_csv(f"{strategy_name}_utilization_std_dev.csv", index=False)
    return df, std_df, failed_pods

def least_requested_priority_score(node, cpu_request=None, memory_request=None, cpu_limit=None, memory_limit=None, alpha=None, beta=None, gamma=None):
    cpu_free = (node.total_cpu - node.used_cpu) / node.total_cpu
    memory_free = (node.total_memory - node.used_memory) / node.total_memory
    return (cpu_free * 10 + memory_free * 10) / 2

def balanced_resource_allocation_score(node, cpu_request=None, memory_request=None, cpu_limit=None, memory_limit=None, alpha=None, beta=None, gamma=None):
    cpu_ratio = node.used_cpu / node.total_cpu
    memory_ratio = node.used_memory / node.total_memory
    return (1 - abs(cpu_ratio - memory_ratio)) * 10

def weighted_residual_resource(node, cpu_request, memory_request, cpu_limit, memory_limit, alpha, beta, gamma):
    available_cpu = node.total_cpu - node.used_cpu
    available_memory = node.total_memory - node.used_memory

    cpu_score = max(min(beta * available_cpu, cpu_limit), cpu_request)
    memory_score = max(min(gamma * available_memory, memory_limit), memory_request)

    return alpha * cpu_score + (1 - alpha) * memory_score

# Simulate scheduling with the modified function
nodes = [Node(10, 10) for _ in range(1)] + [Node(10, 20) for _ in range(1)] + [Node(20, 40) for _ in range(1)] + [Node(20, 80) for _ in range(1)] + [Node(40, 160) for _ in range(1)]
lrp_results, lrp_std_results, lrp_failed_pods = simulate_scheduling(least_requested_priority_score, "Least Requested Priority")
bra_results, bra_std_results, bra_failed_pods = simulate_scheduling(balanced_resource_allocation_score, "Balanced Resource Allocation")
wrr_results, wrr_std_results, wrr_failed_pods = simulate_scheduling(weighted_residual_resource, "Weighted Residual Resource", alpha=0.25, beta=0.96, gamma=0.69)

print(f"Failed pods in Least Requested Priority: {lrp_failed_pods}")
print(f"Failed pods in Balanced Resource Allocation: {bra_failed_pods}")
print(f"Failed pods in Weighted Residual Resource: {wrr_failed_pods}")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))  # Adjust figure size as necessary

# Plot CPU utilization on the first subplot (top-left)
axes[0, 0].plot(lrp_results['Round'], lrp_results['Average CPU Utilization'], label='LRP CPU')
axes[0, 0].plot(bra_results['Round'], bra_results['Average CPU Utilization'], label='BRA CPU')
axes[0, 0].plot(wrr_results['Round'], wrr_results['Average CPU Utilization'], label='WRR CPU')
axes[0, 0].set_title('Average CPU Utilization Over Rounds')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Utilization (%)')
axes[0, 0].legend()

# Plot Memory utilization on the second subplot (top-right)
axes[0, 1].plot(lrp_results['Round'], lrp_results['Average Memory Utilization'], label='LRP Memory')
axes[0, 1].plot(bra_results['Round'], bra_results['Average Memory Utilization'], label='BRA Memory')
axes[0, 1].plot(wrr_results['Round'], wrr_results['Average Memory Utilization'], label='WRR Memory')
axes[0, 1].set_title('Average Memory Utilization Over Rounds')
axes[0, 1].set_xlabel('Round')
axes[0, 1].set_ylabel('Utilization (%)')
axes[0, 1].legend()

# Plot standard deviation of CPU utilization on the third subplot (bottom-left)
axes[1, 0].plot(lrp_std_results['Round'], lrp_std_results['CPU Utilization Std Dev'], label='LRP CPU Std Dev')
axes[1, 0].plot(bra_std_results['Round'], bra_std_results['CPU Utilization Std Dev'], label='BRA CPU Std Dev')
axes[1, 0].plot(wrr_std_results['Round'], wrr_std_results['CPU Utilization Std Dev'], label='WRR CPU Std Dev')
axes[1, 0].set_title('Standard Deviation of CPU Utilization Over Rounds')
axes[1, 0].set_xlabel('Round')
axes[1, 0].set_ylabel('Standard Deviation (%)')
axes[1, 0].legend()

# Plot standard deviation of Memory utilization on the fourth subplot (bottom-right)
axes[1, 1].plot(lrp_std_results['Round'], lrp_std_results['Memory Utilization Std Dev'], label='LRP Memory Std Dev')
axes[1, 1].plot(bra_std_results['Round'], bra_std_results['Memory Utilization Std Dev'], label='BRA Memory Std Dev')
axes[1, 1].plot(wrr_std_results['Round'], wrr_std_results['Memory Utilization Std Dev'], label='WRR Memory Std Dev')
axes[1, 1].set_title('Standard Deviation of Memory Utilization Over Rounds')
axes[1, 1].set_xlabel('Round')
axes[1, 1].set_ylabel('Standard Deviation (%)')
axes[1, 1].legend()

plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()

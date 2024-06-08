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

def simulate_scheduling(scoring_function, strategy_name, alpha=0.25, beta=0.96, gamma=0.69, number_of_pods=500):
    util_data = []
    cpu_stds = []
    memory_stds = []
    failed_pods = 0
    
    print(f"\nSimulating {strategy_name} Strategy with 25 Nodes and {number_of_pods} Pods...")
    random.seed(42)  # for reproducibility

    global nodes
    nodes = [Node(10, 10) for _ in range(5)] + [Node(10, 20) for _ in range(5)] + [Node(20, 40) for _ in range(5)] + [Node(20, 80) for _ in range(5)] + [Node(40, 160) for _ in range(5)]

    for i in range(number_of_pods):
        cpu_request = random.randint(1, 4)
        memory_request = random.randint(1, 16)
        cpu_limit = random.randint(cpu_request, cpu_request + 2)
        memory_limit = random.randint(memory_request, memory_request + 32)
        
        scored_nodes = [(node, scoring_function(node, cpu_request, memory_request, cpu_limit, memory_limit, alpha, beta, gamma), idx) for idx, node in enumerate(nodes)]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        allocated = False # Flag to check if scheduling happens
        for node, score, idx in scored_nodes:
            if node.total_cpu - node.used_cpu >= cpu_request and node.total_memory - node.used_memory >= memory_request:
                actual_cpu = random.randint(cpu_request, cpu_limit)
                actual_memory = random.randint(memory_request, memory_limit)
                if node.total_cpu - node.used_cpu >= actual_cpu and node.total_memory - node.used_memory >= actual_memory:
                    node.allocate_resources(actual_cpu, actual_memory)
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
        avg_std = (cpu_std + memory_std) / 2
        cpu_stds.append(cpu_std)
        memory_stds.append(memory_std)
        util_data.append([i + 1, ave_cpu, ave_memory, avg_std, strategy_name])

    for node in nodes:
        node.reset_resources()

    df = pd.DataFrame(util_data, columns=['Round', 'Average CPU Utilization', 'Average Memory Utilization', 'Average Resource Utilization Std Dev', 'Strategy'])
    return df, failed_pods

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
number_of_nodes = 25
pods_list = [5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000]

improvements_lrp = []
improvements_bra = []

for number_of_pods in pods_list:
    nodes = [Node(10, 10) for _ in range(int(number_of_nodes/5))] + [Node(10, 20) for _ in range(int(number_of_nodes/5))] + [Node(20, 40) for _ in range(int(number_of_nodes/5))] + [Node(20, 80) for _ in range(int(number_of_nodes/5))] + [Node(40, 160) for _ in range(int(number_of_nodes/5))]
    lrp_results, lrp_failed_pods = simulate_scheduling(least_requested_priority_score, "Least Requested Priority", number_of_pods=number_of_pods)
    bra_results, bra_failed_pods = simulate_scheduling(balanced_resource_allocation_score, "Balanced Resource Allocation", number_of_pods=number_of_pods)
    wrr_results, wrr_failed_pods = simulate_scheduling(weighted_residual_resource, "Weighted Residual Resource", alpha=0.25, beta=0.96, gamma=0.69, number_of_pods=number_of_pods)

    print(f"Failed pods in Least Requested Priority: {lrp_failed_pods}")
    print(f"Failed pods in Balanced Resource Allocation: {bra_failed_pods}")
    print(f"Failed pods in Weighted Residual Resource: {wrr_failed_pods}")

    # Calculate percentage improvement
    mean_lrp = lrp_results['Average Resource Utilization Std Dev'].mean()
    mean_bra = bra_results['Average Resource Utilization Std Dev'].mean()
    mean_wrr = wrr_results['Average Resource Utilization Std Dev'].mean()

    improvement_lrp = ((mean_lrp - mean_wrr) / mean_lrp) * 100
    improvement_bra = ((mean_bra - mean_wrr) / mean_bra) * 100

    improvements_lrp.append(improvement_lrp)
    improvements_bra.append(improvement_bra)

    print(f"Percentage improvement of WRR over LRP with {number_of_pods} Pods: {improvement_lrp:.2f}%")
    print(f"Percentage improvement of WRR over BRA with {number_of_pods} Pods: {improvement_bra:.2f}%")

# Plot the improvements
fig, ax = plt.subplots(figsize=(14, 7))  # Adjust figure size as necessary

x = np.arange(len(pods_list))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.barh(x - width/2, improvements_lrp, width, label='Improvement over LRP')
rects2 = ax.barh(x + width/2, improvements_bra, width, label='Improvement over BRA')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Icoming Traffic (Number of Pods)', fontsize=16)
ax.set_xlabel('Improvement (%)', fontsize=16)
ax.set_title('Improvement of WRR over LRP and BRA', fontsize=20)
ax.set_yticks(x)
ax.set_yticklabels(pods_list, fontsize=12)
ax.legend()

# Add labels on top of the bars with smaller font
def add_labels(rects, ax):
    for rect in rects:
        width = rect.get_width()
        ax.annotate(f'{width:.2f}%',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10)

add_labels(rects1, ax)
add_labels(rects2, ax)

fig.tight_layout()

plt.show()

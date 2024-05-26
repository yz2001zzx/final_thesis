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

def simulate_scheduling(scoring_function, strategy_name, alpha=0.5, beta=0.5, gamma=0.5):
    util_data = []
    cpu_stds = []
    memory_stds = []
    failed_pods = 0
    
    print(f"\nSimulating {strategy_name} Strategy...")
    random.seed(42)  # for reproducibility
    for i in range (500):
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

def weighted_residual_resource(node, cpu_request, memory_request, cpu_limit, memory_limit, alpha, beta, gamma):
    available_cpu = node.total_cpu - node.used_cpu
    available_memory = node.total_memory - node.used_memory

    cpu_score = max(min(beta * available_cpu, cpu_limit), cpu_request)
    memory_score = max(min(gamma * available_memory, memory_limit), memory_request)

    return alpha * cpu_score + (1 - alpha) * memory_score

# Simulate scheduling with the modified function
nodes = [Node(10, 10) for _ in range(5)] + [Node(10, 20) for _ in range(5)] + [Node(20, 40) for _ in range(5)] + [Node(20, 80) for _ in range(5)] + [Node(40, 160) for _ in range(5)]

# Vary alpha from 0 to 1 with beta and gamma fixed at 0.5
alpha_results = []
for alpha in np.round(np.arange(0, 1.1, 0.1), 1):
    results, std_results, failed_pods = simulate_scheduling(weighted_residual_resource, f"WRR_alpha_{alpha}", alpha=alpha, beta=0.5, gamma=0.5)
    alpha_results.append((alpha, results, std_results, failed_pods))

# Vary beta from 0 to 1 with alpha and gamma fixed at 0.5
beta_results = []
for beta in np.round(np.arange(0, 1.1, 0.1), 1):
    results, std_results, failed_pods = simulate_scheduling(weighted_residual_resource, f"WRR_beta_{beta}", alpha=0.5, beta=beta, gamma=0.5)
    beta_results.append((beta, results, std_results, failed_pods))

# Vary gamma from 0 to 1 with alpha and beta fixed at 0.5
gamma_results = []
for gamma in np.round(np.arange(0, 1.1, 0.1), 1):
    results, std_results, failed_pods = simulate_scheduling(weighted_residual_resource, f"WRR_gamma_{gamma}", alpha=0.5, beta=0.5, gamma=gamma)
    gamma_results.append((gamma, results, std_results, failed_pods))

# Save data points for alpha, beta, gamma sweeps
for param_results, param_name in zip([alpha_results, beta_results, gamma_results], ['alpha', 'beta', 'gamma']):
    all_results = []
    for param_value, results, std_results, failed_pods in param_results:
        results['Parameter'] = param_value
        std_results['Parameter'] = param_value
        results['Failed Pods'] = failed_pods
        std_results['Failed Pods'] = failed_pods
        all_results.append(results)
        all_results.append(std_results)
    combined_df = pd.concat(all_results)
    combined_df.to_csv(f"WRR_{param_name}_sweep.csv", index=False)

# Function to plot results for a parameter sweep
def plot_results(param_results, param_name):
    greek_letter = {'alpha': r'$\alpha$', 'beta': r'$\beta$', 'gamma': r'$\gamma$'}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))  # Adjust figure size as necessary
    for param_value, results, std_results, _ in param_results:
        label = f'{greek_letter[param_name]} {param_value}'
        axes[0, 0].plot(results['Round'], results['Average CPU Utilization'], label=label)
        axes[0, 1].plot(results['Round'], results['Average Memory Utilization'], label=label)
        axes[1, 0].plot(std_results['Round'], std_results['CPU Utilization Std Dev'], label=label)
        axes[1, 1].plot(std_results['Round'], std_results['Memory Utilization Std Dev'], label=label)
    
    axes[0, 0].set_title(f'Average CPU Utilization Over Rounds ({greek_letter[param_name]} Sweep)')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Utilization (%)')
    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    axes[0, 1].set_title(f'Average Memory Utilization Over Rounds ({greek_letter[param_name]} Sweep)')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Utilization (%)')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    axes[1, 0].set_title(f'Standard Deviation of CPU Utilization Over Rounds ({greek_letter[param_name]} Sweep)')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Standard Deviation (%)')
    axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    axes[1, 1].set_title(f'Standard Deviation of Memory Utilization Over Rounds ({greek_letter[param_name]} Sweep)')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Standard Deviation (%)')
    axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust subplots to fit into figure area and add space at the bottom.
    plt.savefig(f'WRR_{param_name}_sweep.png')
    plt.show()

# Plot results for alpha, beta, gamma sweeps
plot_results(alpha_results, 'alpha')
plot_results(beta_results, 'beta')
plot_results(gamma_results, 'gamma')

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

def simulate_scheduling(scoring_function, strategy_name, rounds, alpha=0.5, beta=0.5, gamma=0.5):
    util_data = []
    cpu_stds = []
    memory_stds = []
    failed_pods = 0
    
    print(f"\nSimulating {strategy_name} Strategy for {rounds} rounds...")
    random.seed(42)  # for reproducibility
    for i in range(rounds):
        cpu_request = random.randint(1, 4)
        memory_request = random.randint(1, 16)
        cpu_limit = random.randint(cpu_request, cpu_request + 2)
        memory_limit = random.randint(memory_request, memory_request + 32)
        
        scored_nodes = [(node, scoring_function(node, cpu_request, memory_request, cpu_limit, memory_limit, alpha, beta, gamma), idx) for idx, node in enumerate(nodes)]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        allocated = False
        for node, score, idx in scored_nodes:
            if node.total_cpu - node.used_cpu >= cpu_request and node.total_memory - node.used_memory >= memory_request:
                actual_cpu = random.randint(cpu_request, cpu_limit)
                actual_memory = random.randint(memory_request, memory_limit)
                if node.total_cpu - node.used_cpu >= actual_cpu and node.total_memory - node.used_memory >= actual_memory:
                    node.allocate_resources(actual_cpu, actual_memory)
                    allocated = True
                    break
        if not allocated:
            failed_pods += 1  # Increment the counter if no node could schedule the pod

        cpu_utilizations = [node.utilization_percentage()[0] for node in nodes]
        memory_utilizations = [node.utilization_percentage()[1] for node in nodes]
        ave_cpu = sum(cpu_utilizations) / len(nodes)
        ave_memory = sum(memory_utilizations) / len(nodes)
        cpu_std = np.std(cpu_utilizations)
        memory_std = np.std(memory_utilizations)
        cpu_stds.append(cpu_std)
        memory_stds.append(memory_std)
        util_data.append([i + 1, ave_cpu, ave_memory])

    for node in nodes:
        node.reset_resources()

    df = pd.DataFrame(util_data, columns=['Round', 'Average CPU Utilization', 'Average Memory Utilization'])
    std_df = pd.DataFrame({
        'Round': range(1, rounds + 1),
        'CPU Utilization Std Dev': cpu_stds,
        'Memory Utilization Std Dev': memory_stds
    })
    return df, std_df, failed_pods

def weighted_residual_resource(node, cpu_request, memory_request, cpu_limit, memory_limit, alpha, beta, gamma):
    available_cpu = node.total_cpu - node.used_cpu
    available_memory = node.total_memory - node.used_memory

    cpu_score = max(min(beta * available_cpu, cpu_limit), cpu_request)
    memory_score = max(min(gamma * available_memory, memory_limit), memory_request)

    return alpha * cpu_score + (1 - alpha) * memory_score

# Function to perform grid search
def grid_search(rounds, step_size=0.1):
    best_params_per_round = []

    for round_count in rounds:
        best_params = None
        best_std = float('inf')
        results = []

        for alpha in np.round(np.arange(0, 1 + step_size, step_size), 1):
            for beta in np.round(np.arange(0, 1 + step_size, step_size), 1):
                for gamma in np.round(np.arange(0, 1 + step_size, step_size), 1):
                    global nodes
                    nodes = [Node(10, 10) for _ in range(5)] + [Node(10, 20) for _ in range(5)] + [Node(20, 40) for _ in range(5)] + [Node(20, 80) for _ in range(5)] + [Node(40, 160) for _ in range(5)]
                    _, std_df, _ = simulate_scheduling(weighted_residual_resource, f"WRR_alpha_{alpha:.1f}_beta_{beta:.1f}_gamma_{gamma:.1f}", round_count, alpha=alpha, beta=beta, gamma=gamma)
                    final_cpu_std = std_df['CPU Utilization Std Dev'].iloc[-1]
                    final_memory_std = std_df['Memory Utilization Std Dev'].iloc[-1]
                    avg_std = (final_cpu_std + final_memory_std) / 2

                    results.append((round_count, alpha, beta, gamma, final_cpu_std, final_memory_std, avg_std))
                    if avg_std < best_std:
                        best_std = avg_std
                        best_params = (alpha, beta, gamma)

        best_params_per_round.append((round_count, best_std, best_params[0], best_params[1], best_params[2]))

    results_df = pd.DataFrame(best_params_per_round, columns=['Round', 'Average Resource Utilization Std Dev', 'Alpha', 'Beta', 'Gamma'])
    results_df.to_csv("grid_search_results.csv", index=False)

    print(results_df)
    return results_df

# Define rounds and perform grid search
rounds = list(range(5, 501, 5))
results_df = grid_search(rounds, step_size=0.1)

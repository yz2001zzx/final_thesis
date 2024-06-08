import random
import math

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

# Initialize nodes
nodes = [Node(5, 5) for _ in range(5)] + \
        [Node(10, 15) for _ in range(5, 10)] + \
        [Node(20, 30) for _ in range(10, 15)] + \
        [Node(40, 5) for _ in range(15, 20)]

# Function to simulate pod scheduling based on a scoring function
def simulate_scheduling(scoring_function, strategy_name):
    print(f"\nSimulating {strategy_name} Strategy...")
    random.seed(42)  # for reproducibility
    for i in range(100):
        cpu_request = random.randint(1, 40)
        memory_request = random.randint(1, 30)
        # Score nodes and sort by best score
        scored_nodes = [(node, scoring_function(node), idx) for idx, node in enumerate(nodes)]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nRound {i + 1} - Pod Request (CPU: {cpu_request}, Memory: {memory_request})")
        print("Ranking of Nodes Before Allocation:")
        for rank, (node, score, idx) in enumerate(scored_nodes):
            print(f"Rank {rank + 1} - Node {idx}: {node}, Score: {score:.2f}")

        # Allocate to the best scored node that fits the request
        for node, score, idx in scored_nodes:
            if node.total_cpu - node.used_cpu >= cpu_request and node.total_memory - node.used_memory >= memory_request:
                print(f"Selecting Node {idx} for Allocation:")
                print(f"Before Allocation - Node {idx}: {node}")
                node.allocate_resources(cpu_request, memory_request)
                remaining_cpu = node.total_cpu - node.used_cpu
                remaining_memory = node.total_memory - node.used_memory
                print(f"After Allocation - Node {idx}: CPU Left: {remaining_cpu}, Memory Left: {remaining_memory}")
                break

    cpu_utilizations = [node.utilization_percentage()[0] for node in nodes]
    memory_utilizations = [node.utilization_percentage()[1] for node in nodes]
    ave_cpu = sum(cpu_utilizations) / len(nodes)
    ave_memory = sum(memory_utilizations) / len(nodes)
    lbi = sum(math.sqrt((cpu - ave_cpu) ** 2 + (mem - ave_memory) ** 2) for cpu, mem in zip(cpu_utilizations, memory_utilizations)) / len(nodes)

    # Reset nodes for next simulation
    for node in nodes:
        node.reset_resources()

    return ave_cpu, ave_memory, lbi

# Scoring functions
def least_requested_priority_score(node):
    cpu_free = (node.total_cpu - node.used_cpu) / node.total_cpu
    memory_free = (node.total_memory - node.used_memory) / node.total_memory
    return (cpu_free * 10 + memory_free * 10) / 2

def balanced_resource_allocation_score(node):
    cpu_ratio = node.used_cpu / node.total_cpu
    memory_ratio = node.used_memory / node.total_memory
    return (1 - abs(cpu_ratio - memory_ratio)) * 10

# Perform simulations and store results
lrp_cpu, lrp_memory, lrp_lbi = simulate_scheduling(least_requested_priority_score, "Least Requested Priority")
bra_cpu, bra_memory, bra_lbi = simulate_scheduling(balanced_resource_allocation_score, "Balanced Resource Allocation")

# Display the results together
print("\nComparison of Strategies:")
print("Least Requested Priority - Average CPU Utilization: {:.2f}%, Average Memory Utilization: {:.2f}%, LBI: {:.2f}".format(lrp_cpu, lrp_memory, lrp_lbi))
print("Balanced Resource Allocation - Average CPU Utilization: {:.2f}%, Average Memory Utilization: {:.2f}%, LBI: {:.2f}".format(bra_cpu, bra_memory, bra_lbi))


#requests to the chatgpt

# Request 1: At the end of each round, print out the average CPU utlization and average memory utilization

# Request 2: Save the average CPU utilization and average memory in each round to an panda dataframe and a csv file

# Request 3: At the end of the round, save the ranking for each node (available CPU, available Memory) and corresponding score to an panda dataframe and a csv file

# Request 4: Make the plot of the average CPU utilization and average memory for different strategies (the X-axis is the round index)
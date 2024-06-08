import matplotlib.pyplot as plt
import numpy as np
import random

# Set the number of rounds
N = 501

# Initialize lists to store the values
cpu_requests = []
memory_requests = []
cpu_limits = []
memory_limits = []

# Simulate the incoming pods
for i in range(N):
    cpu_request = random.randint(1, 4)
    memory_request = random.randint(1, 16)
    cpu_limit = random.randint(cpu_request, cpu_request + 2)
    memory_limit = random.randint(memory_request, memory_request + 32)
    
    cpu_requests.append(cpu_request)
    memory_requests.append(memory_request)
    cpu_limits.append(cpu_limit)
    memory_limits.append(memory_limit)

# Create a range for the X axis
rounds = np.arange(1, N + 1)

# Plot the traffic
plt.figure(figsize=(14, 7))
plt.plot(rounds, cpu_requests, label='CPU Request', linestyle='-', marker='o')
plt.plot(rounds, memory_requests, label='Memory Request', linestyle='-', marker='s')
plt.plot(rounds, cpu_limits, label='CPU Limit', linestyle='--', marker='x')
plt.plot(rounds, memory_limits, label='Memory Limit', linestyle='--', marker='d')

# Set plot labels and title
plt.xlabel('N: The Number of Worker Nodes')
plt.ylabel('Resource')
plt.title('Resource Requests and Limits')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

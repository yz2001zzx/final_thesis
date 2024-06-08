# Final Thesis: A Node Ranking Algorithm for Pod Placement in Kubernetes Clusters

**Author**: Zixiang Zhou (Jacky)

**Thesis Title**: A Node Ranking Algorithm for Pod Placement Considering Varied Pod Characteristics Within the Kubernetes Cluster

**Completion Date**: June 2024

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Logs](#logs)
- [Contributing](#contributing)
- [Contact](#contact)

## Introduction

This repository contains all the code, configuration files, log files, and utilities used for my master thesis titled **"A Node Ranking Algorithm for Pod Placement Considering Varied Pod Characteristics Within the Kubernetes Cluster."** The thesis presents a novel Weighted Residual Resource (WRR) algorithm designed to optimize pod scheduling in Kubernetes by considering various pod characteristics.


## Repository Structure

```plaintext
Testing_Logs/
├── default_scheduler_testing.txt
Utilities/
├── Hypothesis_Testing.py
├── frequency_analysis.py
├── improvement_analysis.py
├── scalability_analysis.py
├── utilization_compare.py
plugins_in_go/
├── WeightedResidualResource.go
├── config.yaml
├── go.mod
├── plugin.go
python_simulator/
├── Bayesian_Optimization.py
├── Grid_Search.py
├── Incoming_traffic.py
├── WRR_Sweep.py
├── grid_search_enhanced.py
├── simulate_default_strategy.py
├── simulate_new_strategy.py
README.md
```

## Requirements

- Python 3.8+
- Kubernetes 1.18+
- Go 1.14+
- Additional Python packages as listed in `requirements.txt`

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install Go dependencies:**
    ```sh
    cd plugins_in_go
    go mod tidy
    ```

## Usage

To run the simulation and the WRR algorithm:

1. **Configure the cluster and scheduler:**
    Modify the configuration files in the `plugins_in_go/config.yaml` directory to suit your cluster setup and scheduling preferences.

2. **Run the Go plugin:**
    ```sh
    cd plugins_in_go
    go run plugin.go
    ```

3. **Run the Python simulations:**
    ```sh
    cd python_simulator
    python simulate_new_strategy.py
    ```

This will execute the WRR algorithm with the provided configurations and log the results in the `Testing_Logs` directory.

## Configuration

- **Cluster Configuration:** Edit `plugins_in_go/config.yaml` to define your cluster setup.
- **Scheduler Configuration:** Adjust the configuration within the same file to customize scheduler settings.

## Logs

Simulation logs and results will be saved in the `Testing_Logs/` directory. Each run will generate a new log file with detailed metrics and performance data.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.


## Contact

For any questions or feedback, please contact me at [jacky.z.zhou@gmail.com].

# gossip-sim

This repository contains simulations for gossip-based distributed algorithms. It currently includes two main functionalities:

1. **Mean Calculation**: Compute various types of means (geometric, harmonic, arithmetic) across distributed nodes
2. **Percentile Estimation**: Allow nodes to determine their relative position (percentile) in a distributed network

Both implementations support various gossip methods and network conditions.

## Features

### Common Features
- **Gossip Methods**: Both implementations include three gossip interaction models:
  - `push-only`: Nodes only send their values to a randomly selected neighbor
  - `pull-only`: Nodes only update their values based on a value received from a randomly selected neighbor
  - `push-pull`: Nodes exchange values with a randomly selected neighbor and both update their values
- **Node Dropouts**: Simulate network failures with configurable probability of node dropouts in each round
  - Supports correlated dropouts where a node's previous state affects its current dropout probability
- **Customizable Simulation**: Configure the number of nodes, iterations, and statistics intervals
- **Reproducibility**: Option to set a random seed for consistent results across runs
- **Interactive Web Interface**: Gradio-based UI for running simulations with real-time visualization

### Mean Calculation Features
- **Different Mean Types**: Supports geometric, harmonic, and arithmetic mean calculations
- **Convergence Tracking**: Monitor how values approach the correct mean over time

### Percentile Estimation Features
- **Two-Phase Algorithm**: 
  1. Node count estimation: Nodes estimate the total number in the network
  2. Index approximation: Nodes determine their relative position (percentile)
- **Order Preservation**: Track how well nodes maintain their correct relative order
- **Error Metrics**: Monitor mean and maximum errors in position estimation

## Installation

The project requires Python 3.x and uses standard libraries along with `numpy`, `matplotlib`, and `gradio`. To set up your environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rishi-s8/gossip-mean.git
   cd gossip-mean
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Mean Calculation

#### Command Line Interface
To run the mean calculation simulation from the command line:
    
```bash
python means/main.py --num_nodes 10000 --num_iterations 50 --task geometric --mode push-pull --stats_interval 5 --seed 42 --dropout_prob 0.1 --dropout_corr 0.5
```

**Command Line Arguments**:
- `--num_nodes`: Number of nodes in the network (default: 10000)
- `--num_iterations`: Number of gossip iterations (default: 50)
- `--task`: Type of mean to calculate (geometric, harmonic, arithmetic). Default is geometric.
- `--mode`: Gossip method to use (push-only, pull-only, push-pull). Default is push-pull.
- `--stats_interval`: Interval of iterations after which to print statistics (default: 10)
- `--seed`: Random seed for reproducibility (default: None)
- `--dropout_prob`: Probability of a node dropping out in a given round (default: 0.0)
- `--dropout_corr`: Correlation factor for dropout probability (default: 0.0)
- `--convergence_rounds`: Stop if values don't change for this many consecutive rounds (default: 0)

#### Interactive Web Interface
To launch the interactive Gradio web interface for mean calculation:

```bash
python means/app.py
```

### Percentile Estimation

#### Command Line Interface
To run the percentile estimation simulation from the command line:
    
```bash
python percentiles/main.py --num_nodes 100 --num_iterations 50 --mode push-pull --stats_interval 5 --seed 42 --dropout_prob 0.1 --dropout_corr 0.5
```

**Command Line Arguments**:
- `--num_nodes`: Number of nodes in the network (default: 100)
- `--num_iterations`: Number of gossip iterations per phase (default: 50)
- `--mode`: Gossip method to use (push-only, pull-only, push-pull). Default is push-pull.
- `--stats_interval`: Interval of iterations after which to print statistics (default: 10)
- `--seed`: Random seed for reproducibility (default: None)
- `--dropout_prob`: Probability of a node dropping out in a given round (default: 0.0)
- `--dropout_corr`: Correlation factor for dropout probability (default: 0.0)
- `--convergence_rounds`: Stop if values don't change for this many consecutive rounds (default: 0)

#### Interactive Web Interface
To launch the interactive Gradio web interface for percentile estimation:

```bash
python percentiles/app.py
```

## Interface Features

Both web interfaces offer:
- Adjustable simulation parameters using sliders and dropdown menus
- Real-time visualization of algorithm convergence
- Detailed statistics and convergence plots
- Predefined examples for quick testing

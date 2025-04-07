# gossip-mean
This repository contains code for simulating different gossip algorithms to compute means (geometric, harmonic, and arithmetic) across a distributed network of nodes. The simulation supports various gossip methods including push-only, pull-only, and push-pull interactions. 

## Features

- **Different Mean Calculations**: Supports geometric, harmonic, and arithmetic mean calculations.
- **Gossip Methods**: Includes three gossip interaction models:
  - `push-only`: Nodes only send their values to a randomly selected neighbor.
  - `pull-only`: Nodes only update their values based on a value received from a randomly selected neighbor.
  - `push-pull`: Nodes exchange values with a randomly selected neighbor and both update their values.
- **Node Dropouts**: Simulate network failures with configurable probability of node dropouts in each round.
  - Supports correlated dropouts where a node's previous state affects its current dropout probability.
- **Customizable Simulation**: Configure the number of nodes, number of iterations, and the frequency of output statistics.
- **Reproducibility**: Option to set a random seed for consistent results across runs.
- **Interactive Web Interface**: A Gradio-based UI for running simulations with real-time visualization.

## Installation

The script requires Python 3.x and uses standard libraries along with `numpy`, `matplotlib`, and `gradio`. To set up your environment, you can follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rishi-s8/gossip-mean.git
   cd gossip-mean
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Usage

### Command Line Interface

To run the simulation from the command line, use:
    
```bash
python main.py --num_nodes 10000 --num_iterations 50 --task geometric --mode push-pull --stats_interval 5 --seed 42 --dropout_prob 0.1 --dropout_corr 0.5
```

### Interactive Web Interface

To launch the interactive Gradio web interface:

```bash
python app.py
```

This will start a local web server and open a browser window where you can:
- Adjust all simulation parameters using sliders and dropdown menus
- Run the simulation with a single click
- View detailed statistics and convergence plots
- Try predefined examples

**Command Line Arguments**:

- --num_nodes: Number of nodes in the network (default: 10000).
- --num_iterations: Number of gossip iterations (default: 50).
- --task: Type of mean to calculate (geometric, harmonic, arithmetic). Default is geometric.
- --mode: Gossip method to use (push-only, pull-only, push-pull). Default is push-pull.
- --stats_interval: Interval of iterations after which to print statistics. If set to 0, no intermediate statistics will be printed (default: 10).
- --seed: Random seed for reproducibility (default: None).
- --dropout_prob: Probability of a node dropping out in a given round (default: 0.0). Value must be between 0.0 and 1.0.
- --dropout_corr: Correlation factor for dropout probability if a node was inactive in the previous round (default: 0.0). Value must be between 0.0 and 1.0.

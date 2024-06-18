# gossip-mean
This repository contains code for simulating different gossip algorithms to compute means (geometric, harmonic, and arithmetic) across a distributed network of nodes. The simulation supports various gossip methods including push-only, pull-only, and push-pull interactions. 

## Features

- **Different Mean Calculations**: Supports geometric, harmonic, and arithmetic mean calculations.
- **Gossip Methods**: Includes three gossip interaction models:
  - `push-only`: Nodes only send their values to a randomly selected neighbor.
  - `pull-only`: Nodes only update their values based on a value received from a randomly selected neighbor.
  - `push-pull`: Nodes exchange values with a randomly selected neighbor and both update their values.
- **Customizable Simulation**: Configure the number of nodes, number of iterations, and the frequency of output statistics.
- **Reproducibility**: Option to set a random seed for consistent results across runs.

## Installation

The script requires Python 3.x and uses standard libraries along with `numpy`. To set up your environment, you can follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rishi-s8/gossip-mean.git
   cd gossip-mean
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Usage

To run the simulation, use the following command:
    
    
    python main.py --num_nodes 10000 --num_iterations 50 --task geometric --mode push-pull --stats_interval 5 --seed 42

**Command Line Arguments**:

- --num_nodes: Number of nodes in the network (default: 10000).
- --num_iterations: Number of gossip iterations (default: 50).
- --task: Type of mean to calculate (geometric, harmonic, arithmetic). Default is geometric.
- --mode: Gossip method to use (push-only, pull-only, push-pull). Default is push-pull.
- --stats_interval: Interval of iterations after which to print statistics. If set to 0, no intermediate statistics will be printed (default: 10).
- --seed: Random seed for reproducibility (default: None).

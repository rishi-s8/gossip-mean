import numpy as np
import random
from statistics import geometric_mean, harmonic_mean, mean
import argparse

class Node:
    def __init__(self, value):
        self.value = value
        self.active = True  # Track if node is active in the current round
        self.prev_active = True  # Track if node was active in the previous round

def perform_gossip_iteration(nodes, update_func, mode, dropout_prob, dropout_corr):
    """Perform a single iteration of the gossip algorithm."""
    # Determine active nodes for this round based on dropout probability and previous state
    for node in nodes:
        node.prev_active = node.active
        if node.prev_active:
            # If node was active in previous round, use regular dropout probability
            node.active = random.random() >= dropout_prob
        else:
            # If node was inactive in previous round, use correlated dropout probability
            node.active = random.random() >= (dropout_prob + dropout_corr * (1 - dropout_prob))
        
    # Get active nodes for this round
    active_nodes = [node for node in nodes if node.active]
    
    if not active_nodes:  # Skip this round if no active nodes
        return False, 0
        
    # Shuffle the list of active nodes to ensure random order of interaction each iteration
    random.shuffle(active_nodes)
    for node in active_nodes:
        # Choose a random active neighbor (not self)
        potential_neighbors = [n for n in active_nodes if n != node]
        if not potential_neighbors:
            continue
            
        neighbor = random.choice(potential_neighbors)

        if mode == 'push-only':
            neighbor.value = update_func(node.value, neighbor.value)
        elif mode == 'pull-only':
            node.value = update_func(neighbor.value, node.value)
        elif mode == 'push-pull':
            new_value = update_func(node.value, neighbor.value)
            node.value, neighbor.value = new_value, new_value
        else:
            raise ValueError("Unsupported gossip type: Choose from 'push-only', 'pull-only', or 'push-pull'")
    
    return True, len(active_nodes)

def gossip(nodes, num_iterations, update_func, task, mode, stats_interval, dropout_prob, dropout_corr):
    print(f"\nGossiping to compute {task.capitalize()} mean using {mode} method...")
    mean_function = get_mean_function(task)
    for iteration in range(num_iterations):
        success, active_count = perform_gossip_iteration(nodes, update_func, mode, dropout_prob, dropout_corr)
        
        if not success:  # Skip this round if no active nodes
            print(f"Warning: No active nodes in iteration {iteration + 1}")
            continue

        if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
            # Calculate statistics over ALL nodes, including inactive ones
            values = [node.value for node in nodes]
            print(f"Stats after {iteration + 1} iterations: Mean = {mean_function(values)}, Std Dev = {np.std(values)}, Active Nodes = {active_count}/{len(nodes)}")

def initialize_nodes(num_nodes):
    return [Node(random.uniform(1, 100)) for _ in range(num_nodes)]

def get_update_func(task):
    if task == "geometric":
        return lambda a, b: np.sqrt(a * b)
    elif task == "harmonic":
        return lambda a, b: (2 * a * b) / (a + b)
    elif task == "arithmetic":
        return lambda a, b: (a + b) / 2
    else:
        raise ValueError("Unsupported task")

def get_mean_function(task):
    if task == "geometric":
        return geometric_mean
    elif task == "harmonic":
        return harmonic_mean
    elif task == "arithmetic":
        return mean
    else:
        raise ValueError("Unsupported task")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gossip Algorithm for Mean Calculation")
    parser.add_argument('--num_nodes', type=int, default=10000, help='Number of nodes in the network')
    parser.add_argument('--num_iterations', type=int, default=50, help='Number of gossip iterations')
    parser.add_argument('--task', type=str, default='geometric', choices=['geometric', 'harmonic', 'arithmetic'], help='Type of mean to calculate')
    parser.add_argument('--mode', type=str, default='push-pull', choices=['push-only', 'pull-only', 'push-pull'], help='Gossip method to use')
    parser.add_argument('--stats_interval', type=int, default=10, help='Interval of iterations after which to print statistics')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--dropout_prob', type=float, default=0.0, help='Probability of a node dropping out in a given round (0.0-1.0)')
    parser.add_argument('--dropout_corr', type=float, default=0.0, help='Correlation factor for dropout probability if a node was inactive in the previous round (0.0-1.0)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    # Set the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate dropout probability and correlation
    if args.dropout_prob < 0.0 or args.dropout_prob > 1.0:
        raise ValueError("Dropout probability must be between 0.0 and 1.0")
    if args.dropout_corr < 0.0 or args.dropout_corr > 1.0:
        raise ValueError("Dropout correlation must be between 0.0 and 1.0")

    nodes = initialize_nodes(args.num_nodes)
    update_func = get_update_func(args.task)
    mean_function = get_mean_function(args.task)

    # Initial values and statistics
    init_values = [node.value for node in nodes]
    print(f"Mean of initial values: {mean_function(init_values)}")
    print(f"Standard deviation of initial values: {np.std(init_values)}\n")

    gossip(nodes, args.num_iterations, update_func, args.task, args.mode, args.stats_interval, args.dropout_prob, args.dropout_corr)

    # Final values and statistics - calculated over ALL nodes, both active and inactive
    final_values = [node.value for node in nodes]
    print("\nCalculating final statistics...")
    print(f"Mean of final values: {mean_function(final_values)}")
    print(f"Standard deviation of final values: {np.std(final_values)}")

if __name__ == "__main__":
    main()
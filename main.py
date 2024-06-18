import numpy as np
import random
from statistics import geometric_mean, harmonic_mean, mean
import argparse

class Node:
    def __init__(self, value):
        self.value = value

def gossip(nodes, num_iterations, update_func, task, mode, stats_interval):
    print(f"\nGossiping to compute {task.capitalize()} mean using {mode} method...")
    mean_function = get_mean_function(task)
    for iteration in range(num_iterations):
        # Shuffle the list of nodes to ensure random order of interaction each iteration
        random.shuffle(nodes)
        for node in nodes:
            neighbor = random.choice(nodes)
            # Ensure the node does not interact with itself
            while neighbor == node:
                neighbor = random.choice(nodes)

            if mode == 'push-only':
                neighbor.value = update_func(node.value, neighbor.value)
            elif mode == 'pull-only':
                node.value = update_func(neighbor.value, node.value)
            elif mode == 'push-pull':
                new_value = update_func(node.value, neighbor.value)
                node.value, neighbor.value = new_value, new_value
            else:
                raise ValueError("Unsupported gossip type: Choose from 'push-only', 'pull-only', or 'push-pull'")

        if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
            values = [node.value for node in nodes]
            print(f"Stats after {iteration + 1} iterations: Mean = {mean_function(values)}, Std Dev = {np.std(values)}")

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
    return parser.parse_args()


def main():
    args = parse_arguments()
    # Set the random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    nodes = initialize_nodes(args.num_nodes)
    update_func = get_update_func(args.task)
    mean_function = get_mean_function(args.task)

    # Initial values and statistics
    init_values = [node.value for node in nodes]
    print(f"Mean of initial values: {mean_function(init_values)}")
    print(f"Standard deviation of initial values: {np.std(init_values)}\n")

    gossip(nodes, args.num_iterations, update_func, args.task, args.mode, args.stats_interval)

    # Final values and statistics
    final_values = [node.value for node in nodes]
    print("\nCalculating final statistics...")
    print(f"Mean of final values: {mean_function(final_values)}")
    print(f"Standard deviation of final values: {np.std(final_values)}")

if __name__ == "__main__":
    main()
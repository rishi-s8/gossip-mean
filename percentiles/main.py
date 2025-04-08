import numpy as np
import random
import argparse
from statistics import mean

# Set global random state
global_random = random.Random()


class Node:
    def __init__(self, index, x_value, is_marked=False):
        # Basic properties
        self.index = index  # Actual index of the node
        self.x = x_value  # The attribute x we're using for comparison

        # For node count estimation (phase 1)
        # Only one node starts with 1.0, rest with 0.0
        self.count_value = 1.0 if is_marked else 0.0

        # For index approximation (phase 2)
        self.random_val = global_random.random()  # Use global random instance
        self.approx_index = None  # Will store the approximated index

        # Activity tracking
        self.active = True
        self.prev_active = True


def perform_count_gossip(nodes, mode, dropout_prob, dropout_corr):
    """Perform gossip to estimate the total number of nodes."""
    # Handle node dropouts
    for node in nodes:
        node.prev_active = node.active
        if node.prev_active:
            node.active = global_random.random() >= dropout_prob
        else:
            node.active = global_random.random() >= (
                dropout_prob + dropout_corr * (1 - dropout_prob)
            )

    active_nodes = [node for node in nodes if node.active]
    if not active_nodes:
        return False, 0

    global_random.shuffle(active_nodes)
    for node in active_nodes:
        potential_neighbors = [n for n in active_nodes if n != node]
        if not potential_neighbors:
            continue

        neighbor = global_random.choice(potential_neighbors)

        # Arithmetic mean for node counting
        if mode == "push-only":
            neighbor.count_value = (node.count_value + neighbor.count_value) / 2
        elif mode == "pull-only":
            node.count_value = (neighbor.count_value + node.count_value) / 2
        elif mode == "push-pull":
            avg = (node.count_value + neighbor.count_value) / 2
            node.count_value = neighbor.count_value = avg

    return True, len(active_nodes)


def perform_index_gossip(nodes, mode, dropout_prob, dropout_corr):
    """Perform gossip to estimate the index of each node."""
    # Handle node dropouts (same as in count gossip)
    for node in nodes:
        node.prev_active = node.active
        if node.prev_active:
            node.active = global_random.random() >= dropout_prob
        else:
            node.active = global_random.random() >= (
                dropout_prob + dropout_corr * (1 - dropout_prob)
            )

    active_nodes = [node for node in nodes if node.active]
    if not active_nodes:
        return False, 0

    global_random.shuffle(active_nodes)
    for node in active_nodes:
        potential_neighbors = [n for n in active_nodes if n != node]
        if not potential_neighbors:
            continue

        neighbor = global_random.choice(potential_neighbors)

        # Handle same random value case (resample)
        if abs(node.random_val - neighbor.random_val) < 1e-10:
            print("Resampling random values due to equality")
            if mode == "push-only":
                neighbor.random_val = global_random.random()
            elif mode == "pull-only":
                node.random_val = global_random.random()
            elif mode == "push-pull":
                if node.index < neighbor.index:
                    node.random_val = global_random.random()
                else:
                    neighbor.random_val = global_random.random()
        # Handle case where r_i < r_j and x_i > x_j (swap random values)
        if (node.random_val < neighbor.random_val and node.x > neighbor.x) or (
            node.random_val > neighbor.random_val and node.x < neighbor.x
        ):
            if mode == "push-only":
                neighbor.random_val, node.random_val = (
                    node.random_val,
                    neighbor.random_val,
                )
            elif mode == "pull-only":
                node.random_val, neighbor.random_val = (
                    neighbor.random_val,
                    node.random_val,
                )
            elif mode == "push-pull":
                node.random_val, neighbor.random_val = (
                    neighbor.random_val,
                    node.random_val,
                )

    return True, len(active_nodes)


def estimate_indices(nodes):
    """Calculate approximate indices for nodes based on their random values and individual node counts."""
    for node in nodes:
        # Each node uses its own count_value to estimate the number of nodes
        node_count_estimate = 1.0 / max(node.count_value, 1e-10)
        # Formula: approx_index = random_val * node_count_estimate
        node.approx_index = node.random_val * node_count_estimate
    return nodes


def check_convergence(prev_values, current_values, epsilon=1e-9):
    """Check if values have converged by comparing previous and current values."""
    if prev_values is None:
        return False

    # Calculate the maximum absolute change across all nodes
    max_change = max(
        abs(prev - curr) for prev, curr in zip(prev_values, current_values)
    )
    return max_change < epsilon


def longest_increasing_subsequence_length(arr):
    """
    Calculate the length of the longest increasing subsequence in an array.
    This represents how many nodes are already in the correct relative order.
    """
    if not arr:
        return 0

    n = len(arr)
    lis = [1] * n

    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j] + 1

    return max(lis)


def gossip_percentile(
    nodes,
    num_iterations,
    mode,
    stats_interval,
    dropout_prob,
    dropout_corr,
    convergence_rounds,
):
    # Phase 1: Estimate the number of nodes
    print("\nPhase 1: Estimating the total number of nodes...")

    # For convergence tracking
    prev_count_values = None
    stable_rounds = 0

    for iteration in range(num_iterations):
        success, active_count = perform_count_gossip(
            nodes, mode, dropout_prob, dropout_corr
        )

        if not success:
            print(f"Warning: No active nodes in iteration {iteration + 1}")
            continue

        # Track current count values for convergence check
        current_count_values = [node.count_value for node in nodes]

        # Check for convergence if convergence_rounds > 0
        if convergence_rounds > 0:
            if check_convergence(prev_count_values, current_count_values):
                stable_rounds += 1
                if stable_rounds >= convergence_rounds:
                    print(
                        f"Phase 1 converged after {iteration + 1} iterations (stable for {convergence_rounds} rounds)."
                    )
                    break
            else:
                stable_rounds = 0

        prev_count_values = current_count_values.copy()

        if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
            count_values = [1.0 / max(node.count_value, 1e-10) for node in nodes]
            print(
                f"Iteration {iteration + 1}: Estimated nodes: {mean(count_values):.2f}, Active: {active_count}/{len(nodes)}"
            )

    # Calculate the estimated number of nodes (for display purposes only)
    final_counts = [1.0 / max(node.count_value, 1e-10) for node in nodes]
    estimated_nodes = mean(final_counts)
    print(f"\nEstimated number of nodes: {estimated_nodes:.2f} (Actual: {len(nodes)})")

    # Phase 2: Approximate indices
    print("\nPhase 2: Approximating node indices...")

    # For convergence tracking for phase 2
    prev_random_values = None
    stable_rounds = 0

    # Define accuracy threshold - how close an estimate needs to be to count as "accurate"
    accuracy_threshold = 1.0

    for iteration in range(num_iterations):
        success, active_count = perform_index_gossip(
            nodes, mode, dropout_prob, dropout_corr
        )

        if not success:
            print(f"Warning: No active nodes in iteration {iteration + 1}")
            continue

        # Track current random values for convergence check
        current_random_values = [node.random_val for node in nodes]

        # Check for convergence if convergence_rounds > 0
        if convergence_rounds > 0:
            if check_convergence(prev_random_values, current_random_values):
                stable_rounds += 1
                if stable_rounds >= convergence_rounds:
                    print(
                        f"Phase 2 converged after {iteration + 1} iterations (stable for {convergence_rounds} rounds)."
                    )
                    break
            else:
                stable_rounds = 0

        prev_random_values = current_random_values.copy()

        if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
            # Calculate current approximate indices to show convergence progress
            temp_nodes = estimate_indices([node for node in nodes])

            # Calculate error metrics to show convergence
            errors = [
                abs(node.index - node.approx_index)
                for node in temp_nodes
                if node.approx_index is not None
            ]

            # Get random values in order of true indices for LIS calculation
            # (First sort nodes by true index)
            sorted_nodes = sorted(nodes, key=lambda x: x.index)
            random_vals_in_true_order = [node.random_val for node in sorted_nodes]

            # Calculate LIS length and express as percentage of total nodes
            lis_length = longest_increasing_subsequence_length(
                random_vals_in_true_order
            )
            lis_percentage = (lis_length / len(nodes)) * 100 if nodes else 0

            if errors:
                mean_error = mean(errors)
                max_error = max(errors)

                print(
                    f"Iteration {iteration + 1}: Active: {active_count}/{len(nodes)}, "
                    f"Mean error: {mean_error:.2f}, Max error: {max_error:.2f}, "
                    f"LIS: {lis_length}/{len(nodes)} ({lis_percentage:.1f}%)"
                )
            else:
                print(
                    f"Iteration {iteration + 1}: Index gossip in progress, Active: {active_count}/{len(nodes)}"
                )

    # Estimate the indices - each node uses its own count estimate
    nodes = estimate_indices(nodes)
    return nodes


def initialize_nodes(num_nodes):
    """Initialize nodes with index based on sorted x values.
    Only one node starts with count_value=1.0, the rest with 0.0."""

    # First, generate all the random x values
    x_values = [global_random.uniform(1, 100) for _ in range(num_nodes)]

    # Sort x values and keep track of the original indices
    sorted_pairs = sorted(enumerate(x_values), key=lambda pair: pair[1])

    # Choose a random node to be marked with value 1.0 for node counting
    marked_node = global_random.randint(0, num_nodes - 1)

    # Create nodes with indices based on the sorted order of x values
    nodes = []
    for sorted_index, (original_index, x_value) in enumerate(sorted_pairs):
        is_marked = sorted_index == marked_node
        nodes.append(Node(sorted_index, x_value, is_marked=is_marked))

    return nodes


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Gossip Algorithm for Index Approximation"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=100, help="Number of nodes in the network"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of gossip iterations per phase",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="push-pull",
        choices=["push-only", "pull-only", "push-pull"],
        help="Gossip method to use",
    )
    parser.add_argument(
        "--stats_interval",
        type=int,
        default=10,
        help="Interval of iterations after which to print statistics",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.0,
        help="Probability of a node dropping out in a given round (0.0-1.0)",
    )
    parser.add_argument(
        "--dropout_corr",
        type=float,
        default=0.0,
        help="Correlation factor for dropout probability if a node was inactive in the previous round (0.0-1.0)",
    )
    parser.add_argument(
        "--convergence_rounds",
        type=int,
        default=0,
        help="Stop if values do not change for this many consecutive rounds. 0 means run for full iterations.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    # Set the random seed for reproducibility - do this FIRST before any random operations
    if args.seed is not None:
        # Set both the global random instance and numpy's random seed
        global_random.seed(args.seed)
        random.seed(args.seed)  # Also set Python's built-in random
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Validate dropout parameters
    if args.dropout_prob < 0.0 or args.dropout_prob > 1.0:
        raise ValueError("Dropout probability must be between 0.0 and 1.0")
    if args.dropout_corr < 0.0 or args.dropout_corr > 1.0:
        raise ValueError("Dropout correlation must be between 0.0 and 1.0")
    if args.convergence_rounds < 0:
        raise ValueError("Convergence rounds must be non-negative")

    # Initialize nodes
    nodes = initialize_nodes(args.num_nodes)

    # Run the gossip algorithm for index approximation
    nodes = gossip_percentile(
        nodes,
        args.num_iterations,
        args.mode,
        args.stats_interval,
        args.dropout_prob,
        args.dropout_corr,
        args.convergence_rounds,
    )

    # Print results for a sample of nodes
    print("\nResults for a sample of nodes:")
    print("Actual Index | Approximate Index | X Value | Random Value")
    print("--------------------------------------------------")
    sample_size = min(10, args.num_nodes)
    sample_indices = sorted(random.sample(range(args.num_nodes), sample_size))
    for idx in sample_indices:
        node = nodes[idx]
        print(
            f"{node.index:12d} | {node.approx_index:16.2f} | {node.x:7.2f} | {node.random_val:.6f}"
        )

    # Calculate error metrics
    errors = [abs(node.index - node.approx_index) for node in nodes]
    print(f"\nMean absolute error: {mean(errors):.2f}")
    print(f"Max absolute error: {max(errors):.2f}")


if __name__ == "__main__":
    main()

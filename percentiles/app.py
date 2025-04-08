import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mean
import time

# Import from percentiles.py
from percentiles import (
    Node,
    initialize_nodes,
    perform_count_gossip,
    perform_index_gossip,
    estimate_indices,
    check_convergence,
    longest_increasing_subsequence_length,
)

# Set global random state
global_random = random.Random()


def run_simulation(
    num_nodes,
    num_iterations,
    mode,
    stats_interval,
    seed,
    dropout_prob,
    dropout_corr,
    convergence_rounds,
):
    # Set random seed if provided - ensure this happens FIRST
    if seed is not None:
        global_random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")

    # Initialize nodes
    nodes = initialize_nodes(num_nodes)

    # Initial statistics
    logs = [
        f"Starting simulation with {num_nodes} nodes...",
    ]
    status = "Initializing simulation..."

    # Create plotting components with 6 subplots instead of 5
    fig = plt.figure(figsize=(14, 30))
    gs = fig.add_gridspec(6, 1, hspace=0.5)

    # First subplot: Node count estimation
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Estimated Node Count")
    ax1.set_title("Node Count Estimation (Phase 1)")
    ax1.grid(True)
    ax1.axhline(y=num_nodes, color='r', linestyle='--', alpha=0.7, label="Actual Count")
    ax1.legend()

    # Second subplot: Active nodes in Phase 1
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Number of Active Nodes")
    ax2.set_title("Active Nodes (Phase 1)")
    ax2.grid(True)

    # Third subplot: Index approximation error
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Mean Absolute Error")
    ax3.set_title("Index Approximation Error (Phase 2)")
    ax3.grid(True)

    # Fourth subplot: Active nodes in Phase 2
    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlabel("Iterations")
    ax4.set_ylabel("Number of Active Nodes")
    ax4.set_title("Active Nodes (Phase 2)")
    ax4.grid(True)

    # Fifth subplot: Longest Increasing Subsequence percentage (full view)
    ax5 = fig.add_subplot(gs[4])
    ax5.set_xlabel("Iterations")
    ax5.set_ylabel("LIS Percentage")
    ax5.set_title("Longest Increasing Subsequence (% of nodes in correct order)")
    ax5.grid(True)
    ax5.set_ylim(0, 100)
    
    # Sixth subplot: Zoomed-in LIS percentage view
    ax6 = fig.add_subplot(gs[5])
    ax6.set_xlabel("Iterations")
    ax6.set_ylabel("LIS Percentage")
    ax6.set_title("Zoomed View: Last 50% of Iterations")
    ax6.grid(True)

    # Data for tracking over iterations
    phase1_iterations = []
    phase1_estimates = []
    phase1_active = []
    
    phase2_iterations = []
    phase2_errors = []
    phase2_active = []
    phase2_lis = []

    # Initial plot
    yield "\n".join(logs), status, fig

    # Phase 1: Estimate the number of nodes
    logs.append("\nPhase 1: Estimating the total number of nodes...")
    status = "Running node count estimation (Phase 1)..."

    prev_count_values = None
    stable_rounds = 0

    batch_size = min(5, stats_interval)
    total_batches = int(np.ceil(num_iterations / batch_size))

    for batch in range(total_batches):
        start_iter = batch * batch_size
        end_iter = min((batch + 1) * batch_size, num_iterations)

        for iteration in range(start_iter, end_iter):
            success, active_count = perform_count_gossip(
                nodes, mode, dropout_prob, dropout_corr
            )

            if not success:
                logs.append(f"Warning: No active nodes in iteration {iteration + 1}")
                continue

            # Track current count values for convergence check
            current_count_values = [node.count_value for node in nodes]

            # Check for convergence
            if convergence_rounds > 0:
                if check_convergence(prev_count_values, current_count_values):
                    stable_rounds += 1
                    if stable_rounds >= convergence_rounds:
                        logs.append(
                            f"Phase 1 converged after {iteration + 1} iterations (stable for {convergence_rounds} rounds)."
                        )
                        end_iter = iteration + 1
                        break
                else:
                    stable_rounds = 0

            prev_count_values = current_count_values.copy()

            # Update statistics at each interval
            if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
                count_estimates = [1.0 / max(node.count_value, 1e-10) for node in nodes]
                avg_estimate = mean(count_estimates)
                
                logs.append(
                    f"Iteration {iteration + 1}: Estimated nodes: {avg_estimate:.2f}, "
                    f"Active: {active_count}/{num_nodes}"
                )

                # Store for plotting
                phase1_iterations.append(iteration + 1)
                phase1_estimates.append(avg_estimate)
                phase1_active.append(active_count)

        # Update plot after batch
        if batch % 2 == 0 or batch == total_batches - 1:
            # Add current data point if not already added
            if not phase1_iterations or phase1_iterations[-1] < end_iter:
                count_estimates = [1.0 / max(node.count_value, 1e-10) for node in nodes]
                avg_estimate = mean(count_estimates)
                active_count = sum(1 for node in nodes if node.active)
                
                phase1_iterations.append(end_iter)
                phase1_estimates.append(avg_estimate)
                phase1_active.append(active_count)
            
            # Update plot data
            ax1.clear()
            ax1.plot(phase1_iterations, phase1_estimates, 'b-o', label="Estimated Count")
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel("Estimated Node Count")
            ax1.set_title("Node Count Estimation (Phase 1)")
            ax1.grid(True)
            ax1.axhline(y=num_nodes, color='r', linestyle='--', alpha=0.7, label="Actual Count")
            ax1.legend()
            
            # Update active nodes plot separately
            ax2.clear()
            ax2.plot(phase1_iterations, phase1_active, 'g-o', label="Active Nodes")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Number of Active Nodes")
            ax2.set_title("Active Nodes (Phase 1)")
            ax2.grid(True)
            ax2.legend()
            
            # Return updated figure
            yield "\n".join(logs), status, fig

    # Calculate the final estimated number of nodes
    final_counts = [1.0 / max(node.count_value, 1e-10) for node in nodes]
    estimated_nodes = mean(final_counts)
    logs.append(f"\nEstimated number of nodes: {estimated_nodes:.2f} (Actual: {num_nodes})")

    # Phase 2: Approximate indices
    logs.append("\nPhase 2: Approximating node indices...")
    status = "Running index approximation (Phase 2)..."

    # For convergence tracking
    prev_random_values = None
    stable_rounds = 0

    for batch in range(total_batches):
        start_iter = batch * batch_size
        end_iter = min((batch + 1) * batch_size, num_iterations)

        for iteration in range(start_iter, end_iter):
            success, active_count = perform_index_gossip(
                nodes, mode, dropout_prob, dropout_corr
            )

            if not success:
                logs.append(f"Warning: No active nodes in iteration {iteration + 1}")
                continue

            # Track current random values for convergence check
            current_random_values = [node.random_val for node in nodes]

            # Check for convergence
            if convergence_rounds > 0:
                if check_convergence(prev_random_values, current_random_values):
                    stable_rounds += 1
                    if stable_rounds >= convergence_rounds:
                        logs.append(
                            f"Phase 2 converged after {iteration + 1} iterations (stable for {convergence_rounds} rounds)."
                        )
                        end_iter = iteration + 1
                        break
                else:
                    stable_rounds = 0

            prev_random_values = current_random_values.copy()

            # Update statistics at each interval
            if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
                # Calculate current approximate indices to show convergence progress
                temp_nodes = estimate_indices([node for node in nodes])
                
                # Calculate error metrics
                errors = [abs(node.index - node.approx_index) for node in temp_nodes if node.approx_index is not None]
                
                # Get random values in order of true indices for LIS calculation
                sorted_nodes = sorted(nodes, key=lambda x: x.index)
                random_vals_in_true_order = [node.random_val for node in sorted_nodes]
                
                # Calculate LIS length and percentage
                lis_length = longest_increasing_subsequence_length(random_vals_in_true_order)
                lis_percentage = (lis_length / len(nodes)) * 100 if nodes else 0
                
                if errors:
                    mean_error = mean(errors)
                    logs.append(
                        f"Iteration {iteration + 1}: Active: {active_count}/{num_nodes}, "
                        f"Mean error: {mean_error:.2f}, LIS: {lis_percentage:.1f}%"
                    )
                    
                    # Store for plotting
                    phase2_iterations.append(iteration + 1)
                    phase2_errors.append(mean_error)
                    phase2_active.append(active_count)
                    phase2_lis.append(lis_percentage)
                else:
                    logs.append(
                        f"Iteration {iteration + 1}: Index gossip in progress, Active: {active_count}/{num_nodes}"
                    )

        # Update plot after batch
        if batch % 2 == 0 or batch == total_batches - 1:
            # Add current data point if available and not already added
            if phase2_errors and (not phase2_iterations or phase2_iterations[-1] < end_iter):
                temp_nodes = estimate_indices([node for node in nodes])
                errors = [abs(node.index - node.approx_index) for node in temp_nodes if node.approx_index is not None]
                
                # Get LIS percentage
                sorted_nodes = sorted(nodes, key=lambda x: x.index)
                random_vals_in_true_order = [node.random_val for node in sorted_nodes]
                lis_length = longest_increasing_subsequence_length(random_vals_in_true_order)
                lis_percentage = (lis_length / len(nodes)) * 100 if nodes else 0
                
                if errors:
                    mean_error = mean(errors)
                    active_count = sum(1 for node in nodes if node.active)
                    
                    phase2_iterations.append(end_iter)
                    phase2_errors.append(mean_error)
                    phase2_active.append(active_count)
                    phase2_lis.append(lis_percentage)
            
            # Update error plot
            if phase2_errors:
                ax3.clear()
                ax3.plot(phase2_iterations, phase2_errors, 'r-o', label="Mean Absolute Error")
                ax3.set_xlabel("Iterations")
                ax3.set_ylabel("Mean Absolute Error")
                ax3.set_title("Index Approximation Error (Phase 2)")
                ax3.grid(True)
                ax3.legend()
                
                # Update active nodes plot separately
                ax4.clear()
                ax4.plot(phase2_iterations, phase2_active, 'g-o', label="Active Nodes")
                ax4.set_xlabel("Iterations")
                ax4.set_ylabel("Number of Active Nodes")
                ax4.set_title("Active Nodes (Phase 2)")
                ax4.grid(True)
                ax4.legend()
                
                # Update LIS plot (full)
                ax5.clear()
                ax5.plot(phase2_iterations, phase2_lis, '-o', color='purple', label="LIS Percentage")
                ax5.set_xlabel("Iterations")
                ax5.set_ylabel("LIS Percentage (%)")
                ax5.set_title("Longest Increasing Subsequence (% of nodes in correct order)")
                ax5.grid(True)
                ax5.set_ylim(0, 100)
                ax5.legend()
                
                # Add zoomed-in version for last half of iterations
                if len(phase2_iterations) > 1:
                    # Clear the zoomed plot (which now has its own dedicated axes)
                    ax6.clear()
                    
                    # Only show the last half of data
                    half_idx = max(1, len(phase2_iterations) // 2)
                    zoomed_iterations = phase2_iterations[half_idx:]
                    zoomed_lis = phase2_lis[half_idx:]
                    
                    if zoomed_iterations:
                        # Calculate best y-axis limits to focus on the data
                        min_lis = min(zoomed_lis) * 0.98 if zoomed_lis else 0
                        max_lis = max(min(100, max(zoomed_lis) * 1.02), min_lis + 5) if zoomed_lis else 100
                        
                        ax6.plot(zoomed_iterations, zoomed_lis, '-o', color='purple', label="LIS % (Last 50%)")
                        ax6.set_xlabel("Iterations")
                        ax6.set_ylabel("LIS Percentage (%)")
                        ax6.set_title("Zoomed View: Last 50% of Iterations")
                        ax6.grid(True)
                        ax6.set_ylim(min_lis, max_lis)
                        ax6.legend()
        
            # Return updated figure
            yield "\n".join(logs), status, fig

    # Estimate final indices
    nodes = estimate_indices(nodes)
    
    # Calculate final error metrics
    errors = [abs(node.index - node.approx_index) for node in nodes]
    mean_absolute_error = mean(errors)
    max_error = max(errors)
    
    # Calculate final LIS
    sorted_nodes = sorted(nodes, key=lambda x: x.index)
    random_vals_in_true_order = [node.random_val for node in sorted_nodes]
    lis_length = longest_increasing_subsequence_length(random_vals_in_true_order)
    lis_percentage = (lis_length / len(nodes)) * 100
    
    # Show final results
    logs.append("\nFinal results:")
    logs.append(f"Mean absolute error: {mean_absolute_error:.2f}")
    logs.append(f"Max error: {max_error:.2f}")
    logs.append(f"Nodes in correct relative order: {lis_length}/{num_nodes} ({lis_percentage:.1f}%)")
    
    # Print results for a sample of nodes
    logs.append("\nResults for a sample of nodes:")
    logs.append("Actual Index | Approximate Index | X Value | Random Value")
    logs.append("--------------------------------------------------")
    sample_size = min(10, num_nodes)
    sample_indices = sorted(random.sample(range(num_nodes), sample_size))
    for idx in sample_indices:
        node = nodes[idx]
        logs.append(
            f"{node.index:12d} | {node.approx_index:16.2f} | {node.x:7.2f} | {node.random_val:.6f}"
        )
    
    status = "Simulation complete"
    yield "\n".join(logs), status, fig


# Create the Gradio interface
with gr.Blocks(title="Gossip-based Percentile Estimation") as demo:
    gr.Markdown("# Gossip-based Percentile Estimation")
    gr.Markdown(
        """
    Simulate gossip algorithms to estimate node indices (percentiles) in a distributed network.
    The simulation has two phases:
    1. **Node count estimation**: Nodes estimate the total number of nodes in the network.
    2. **Index approximation**: Nodes approximate their relative position (percentile) in the network.
    """
    )

    with gr.Row():
        with gr.Column():
            # Input components
            num_nodes = gr.Slider(
                minimum=10, maximum=1000, value=100, step=10, label="Number of Nodes"
            )
            num_iterations = gr.Slider(
                minimum=10, maximum=100, value=50, step=5, label="Number of Iterations per Phase"
            )
            mode = gr.Radio(
                ["push-only", "pull-only", "push-pull"],
                value="push-pull",
                label="Gossip Method",
            )
            stats_interval = gr.Slider(
                minimum=1, maximum=20, value=5, step=1, label="Statistics Interval"
            )
            seed = gr.Number(value=None, label="Random Seed (optional)", precision=0)
            dropout_prob = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                label="Dropout Probability",
            )
            dropout_corr = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                label="Dropout Correlation",
            )
            convergence_rounds = gr.Slider(
                minimum=0,
                maximum=20,
                value=0,
                step=1,
                label="Convergence Rounds (0 to disable)",
            )

            run_button = gr.Button("Run Simulation")

        with gr.Column():
            # Output components
            with gr.Column(variant="panel"):
                gr.Markdown("### Simulation Progress")
                output_text = gr.Textbox(label="Results Log", lines=10)

            with gr.Column(variant="panel"):
                gr.Markdown("### Visualization")
                status_text = gr.Textbox(label="Current Status", lines=1, max_lines=1)
                output_plot = gr.Plot(label="Convergence Plots")

    # Connect the run button to the simulation function
    run_button.click(
        fn=run_simulation,
        inputs=[
            num_nodes,
            num_iterations,
            mode,
            stats_interval,
            seed,
            dropout_prob,
            dropout_corr,
            convergence_rounds,
        ],
        outputs=[output_text, status_text, output_plot],
    )

    # Add examples for quick testing
    gr.Examples(
        examples=[
            [100, 50, "push-pull", 5, 42, 0.0, 0.0, 3],
            [100, 50, "push-pull", 5, 42, 0.2, 0.5, 0],
            [50, 75, "push-only", 10, 123, 0.1, 0.3, 5],
        ],
        inputs=[
            num_nodes,
            num_iterations,
            mode,
            stats_interval,
            seed,
            dropout_prob,
            dropout_corr,
            convergence_rounds,
        ],
    )

if __name__ == "__main__":
    # Set default seed for reproducibility
    random.seed(42)
    global_random.seed(42)
    np.random.seed(42)
    
    demo.launch()

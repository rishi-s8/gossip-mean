import gradio as gr
import numpy as np
import plotly.graph_objects as go
import random
from statistics import mean

# Import from main.py instead of percentiles.py
from main import (
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

    # Initialize nodes
    nodes = initialize_nodes(num_nodes)

    # Initial statistics
    logs = [
        f"Starting simulation with {num_nodes} nodes...",
    ]
    status = "Initializing simulation..."

    # Create individual Plotly figures instead of matplotlib subplots
    fig1 = go.Figure()  # Node count estimation
    fig2 = go.Figure()  # Active nodes in Phase 1
    fig3 = go.Figure()  # Index approximation error
    fig4 = go.Figure()  # Active nodes in Phase 2
    fig5 = go.Figure()  # Longest Increasing Subsequence percentage (full view)
    fig6 = go.Figure()  # Zoomed-in LIS percentage view

    # First figure: Node count estimation
    fig1.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Estimated Count",
            line=dict(color="blue"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Actual Count",
            line=dict(color="red", dash="dash"),
        )
    )
    fig1.update_layout(
        title="Node Count Estimation (Phase 1)",
        xaxis_title="Iterations",
        yaxis_title="Estimated Node Count",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Second figure: Active nodes in Phase 1
    fig2.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Active Nodes",
            line=dict(color="green"),
        )
    )
    fig2.update_layout(
        title="Active Nodes (Phase 1)",
        xaxis_title="Iterations",
        yaxis_title="Number of Active Nodes",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Third figure: Index approximation error
    fig3.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Mean Absolute Error",
            line=dict(color="red"),
        )
    )
    fig3.update_layout(
        title="Index Approximation Error (Phase 2)",
        xaxis_title="Iterations",
        yaxis_title="Mean Absolute Error",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Fourth figure: Active nodes in Phase 2
    fig4.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Active Nodes",
            line=dict(color="green"),
        )
    )
    fig4.update_layout(
        title="Active Nodes (Phase 2)",
        xaxis_title="Iterations",
        yaxis_title="Number of Active Nodes",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Fifth figure: Longest Increasing Subsequence percentage
    fig5.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="LIS Percentage",
            line=dict(color="purple"),
        )
    )
    fig5.update_layout(
        title="Longest Increasing Subsequence (% of nodes in correct order)",
        xaxis_title="Iterations",
        yaxis_title="LIS Percentage (%)",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Sixth figure: Zoomed-in LIS percentage view
    fig6.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="LIS % (Last 50%)",
            line=dict(color="purple"),
        )
    )
    fig6.update_layout(
        title="Zoomed View: Last 50% of Iterations",
        xaxis_title="Iterations",
        yaxis_title="LIS Percentage (%)",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Add grid to all plots
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

    # Create a list of figures
    figures = [fig1, fig2, fig3, fig4, fig5, fig6]

    # Data for tracking over iterations
    phase1_iterations = []
    phase1_estimates = []
    phase1_active = []

    phase2_iterations = []
    phase2_errors = []
    phase2_active = []
    phase2_lis = []

    # Initial plot
    yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
        3
    ], figures[4], figures[5]

    # Phase 1: Estimate the number of nodes
    logs.append("\nPhase 1: Estimating the total number of nodes...")
    status = "Running node count estimation (Phase 1)..."

    prev_count_values = None
    stable_rounds = 0
    converged = False  # Add flag to track if convergence has been reached

    batch_size = min(5, stats_interval)
    total_batches = int(np.ceil(num_iterations / batch_size))

    for batch in range(total_batches):
        if converged:  # Check if we've already converged and skip remaining batches
            break

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
                        converged = True  # Set the convergence flag
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

            # Update plots for Phase 1
            fig1.data[0].x = phase1_iterations
            fig1.data[0].y = phase1_estimates
            fig1.data[1].x = [
                phase1_iterations[0] if phase1_iterations else 0,
                phase1_iterations[-1] if phase1_iterations else num_iterations,
            ]
            fig1.data[1].y = [num_nodes, num_nodes]

            fig2.data[0].x = phase1_iterations
            fig2.data[0].y = phase1_active

            # Return updated figures
            yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
                3
            ], figures[4], figures[5]

    # Calculate the final estimated number of nodes
    final_counts = [1.0 / max(node.count_value, 1e-10) for node in nodes]
    estimated_nodes = mean(final_counts)
    logs.append(
        f"\nEstimated number of nodes: {estimated_nodes:.2f} (Actual: {num_nodes})"
    )

    # Phase 2: Approximate indices
    logs.append("\nPhase 2: Approximating node indices...")
    status = "Running index approximation (Phase 2)..."

    # For convergence tracking
    prev_random_values = None
    stable_rounds = 0
    converged = False  # Reset convergence flag for Phase 2

    for batch in range(total_batches):
        if converged:  # Check if we've already converged and skip remaining batches
            break

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
                        converged = True  # Set the convergence flag
                        break
                else:
                    stable_rounds = 0

            prev_random_values = current_random_values.copy()

            # Update statistics at each interval
            if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
                # Calculate current approximate indices to show convergence progress
                temp_nodes = estimate_indices([node for node in nodes])

                # Calculate error metrics
                errors = [
                    abs(node.index - node.approx_index)
                    for node in temp_nodes
                    if node.approx_index is not None
                ]

                # Get random values in order of true indices for LIS calculation
                sorted_nodes = sorted(nodes, key=lambda x: x.index)
                random_vals_in_true_order = [node.random_val for node in sorted_nodes]

                # Calculate LIS length and percentage
                lis_length = longest_increasing_subsequence_length(
                    random_vals_in_true_order
                )
                lis_percentage = (lis_length / len(nodes)) * 100 if nodes else 0

                if errors:
                    mean_error = mean(errors)
                    logs.append(
                        f"Iteration {iteration + 1}: Active: {active_count}/{num_nodes}, "
                        f"Mean error: {mean_error:.2f}, LIS: {lis_length}/{len(nodes)} ({lis_percentage:.1f}%)"
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
            if phase2_errors and (
                not phase2_iterations or phase2_iterations[-1] < end_iter
            ):
                temp_nodes = estimate_indices([node for node in nodes])
                errors = [
                    abs(node.index - node.approx_index)
                    for node in temp_nodes
                    if node.approx_index is not None
                ]

                # Get LIS percentage
                sorted_nodes = sorted(nodes, key=lambda x: x.index)
                random_vals_in_true_order = [node.random_val for node in sorted_nodes]
                lis_length = longest_increasing_subsequence_length(
                    random_vals_in_true_order
                )
                lis_percentage = (lis_length / len(nodes)) * 100 if nodes else 0

                if errors:
                    mean_error = mean(errors)
                    active_count = sum(1 for node in nodes if node.active)

                    phase2_iterations.append(end_iter)
                    phase2_errors.append(mean_error)
                    phase2_active.append(active_count)
                    phase2_lis.append(lis_percentage)

            # Update Phase 2 plots if we have data
            if phase2_errors:
                # Update error plot
                fig3.data[0].x = phase2_iterations
                fig3.data[0].y = phase2_errors

                # Update active nodes in phase 2
                fig4.data[0].x = phase2_iterations
                fig4.data[0].y = phase2_active

                # Update LIS plot (full view)
                fig5.data[0].x = phase2_iterations
                fig5.data[0].y = phase2_lis

                # Update zoomed LIS plot (last 50%)
                if len(phase2_iterations) > 1:
                    half_idx = max(1, len(phase2_iterations) // 2)
                    zoomed_iterations = phase2_iterations[half_idx:]
                    zoomed_lis = phase2_lis[half_idx:]

                    if zoomed_iterations:
                        # Calculate best y-axis limits to focus on the data
                        min_lis = min(zoomed_lis) * 0.98 if zoomed_lis else 0
                        max_lis = (
                            max(min(100, max(zoomed_lis) * 1.02), min_lis + 5)
                            if zoomed_lis
                            else 100
                        )

                        fig6.data[0].x = zoomed_iterations
                        fig6.data[0].y = zoomed_lis
                        fig6.update_yaxes(range=[min_lis, max_lis])

            # Return updated figures
            yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
                3
            ], figures[4], figures[5]

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
    logs.append(
        f"Nodes in correct relative order: {lis_length}/{num_nodes} ({lis_percentage:.1f}%)"
    )

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
    yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
        3
    ], figures[4], figures[5]


# Create the Gradio interface
with gr.Blocks(
    title="Gossip-based Percentile Estimation",
    css="""
    .plot-container { 
        width: 100% !important; 
        min-height: 350px !important; 
        display: block !important;
        margin-bottom: 20px !important;
    }
    /* Fix for the container height */
    .gradio-container {
        max-width: 100% !important;
    }
""",
) as demo:
    gr.Markdown("# Gossip-based Percentile Estimation")
    gr.Markdown(
        """
    Simulate gossip algorithms to estimate node indices (percentiles) in a distributed network.
    The simulation has two phases:
    1. **Node count estimation**: Nodes estimate the total number of nodes in the network.
    2. **Index approximation**: Nodes approximate their relative position (percentile) in the network.
    """
    )

    # Input controls section
    with gr.Row():
        # Left column for parameters
        with gr.Column(scale=2):
            num_nodes = gr.Slider(
                minimum=10, maximum=10000, value=100, step=10, label="Number of Nodes"
            )
            num_iterations = gr.Slider(
                minimum=10,
                maximum=5000,
                value=50,
                step=5,
                label="Number of Iterations per Phase",
            )
            mode = gr.Radio(
                ["push-only", "pull-only", "push-pull"],
                value="push-pull",
                label="Gossip Method",
            )
            stats_interval = gr.Slider(
                minimum=1, maximum=100, value=5, step=1, label="Statistics Interval"
            )

        # Right column for more parameters
        with gr.Column(scale=2):
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
                maximum=1000,
                value=0,
                step=1,
                label="Convergence Rounds (0 to disable)",
            )

    # Run button in its own row
    with gr.Row():
        run_button = gr.Button("Run Simulation", size="large")

    # Status display
    with gr.Row():
        status_text = gr.Textbox(label="Current Status", lines=1, max_lines=1)

    # Full-width visualization section
    with gr.Row():
        with gr.Column(scale=1, min_width=800):
            gr.Markdown("### Visualization")
            # Create separate plot components for each figure
            output_plot1 = gr.Plot(
                label="Node Count Estimation (Phase 1)", elem_classes="plot-container"
            )
            output_plot2 = gr.Plot(
                label="Active Nodes (Phase 1)", elem_classes="plot-container"
            )
            output_plot3 = gr.Plot(
                label="Index Approximation Error (Phase 2)",
                elem_classes="plot-container",
            )
            output_plot4 = gr.Plot(
                label="Active Nodes (Phase 2)", elem_classes="plot-container"
            )
            output_plot5 = gr.Plot(
                label="Longest Increasing Subsequence", elem_classes="plot-container"
            )
            output_plot6 = gr.Plot(
                label="Zoomed View: Last 50% of Iterations",
                elem_classes="plot-container",
            )

    # Results log at the bottom
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Simulation Progress")
            output_text = gr.Textbox(label="Results Log", lines=8)

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
        outputs=[
            output_text,
            status_text,
            output_plot1,
            output_plot2,
            output_plot3,
            output_plot4,
            output_plot5,
            output_plot6,
        ],
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

    # Launch the demo without CSS parameter
    demo.launch()

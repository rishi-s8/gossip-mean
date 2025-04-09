import gradio as gr
import numpy as np
import plotly.graph_objects as go
import random

# Import from main.py
from main import (
    initialize_nodes,
    get_update_func,
    get_mean_function,
    perform_gossip_iteration,
)

# Set global random state
global_random = random.Random()


def check_convergence(prev_values, current_values, epsilon=1e-9):
    """Check if values have converged by comparing previous and current values."""
    if prev_values is None:
        return False

    # Calculate the maximum absolute change across all nodes
    max_change = max(
        abs(prev - curr) for prev, curr in zip(prev_values, current_values)
    )
    return max_change < epsilon


def run_simulation(
    num_nodes,
    num_iterations,
    task,
    mode,
    stats_interval,
    seed,
    dropout_prob,
    dropout_corr,
    convergence_rounds,
):
    # Set random seed if provided - ensure this happens FIRST
    if seed is not None:
        # Set both the global random instance and numpy's random seed
        global_random.seed(seed)
        random.seed(seed)  # Also set Python's built-in random
        np.random.seed(seed)

    # Initialize nodes and functions
    nodes = initialize_nodes(num_nodes)
    update_func = get_update_func(task)
    mean_function = get_mean_function(task)

    # Initial statistics
    init_values = [node.value for node in nodes]
    init_mean = mean_function(init_values)
    init_std = np.std(init_values)

    # For tracking statistics over iterations
    all_means = [init_mean]
    all_stds = [init_std]
    active_nodes_counts = [num_nodes]
    iterations = [0]

    # Store all values at each stats interval for box plots
    all_values_at_intervals = [init_values]

    # Start simulation
    logs = [
        f"Starting simulation with {num_nodes} nodes...",
        f"Initial mean: {init_mean:.4f}, Initial std dev: {init_std:.4f}",
    ]

    status = "Initializing simulation..."

    # Create separate figures for each visualization instead of subplots
    fig1 = go.Figure()  # Full convergence plot
    fig2 = go.Figure()  # Zoomed convergence plot
    fig3 = go.Figure()  # Standard deviation (linear)
    fig4 = go.Figure()  # Standard deviation (log)
    fig5 = go.Figure()  # Active nodes

    # First figure: Full convergence plot
    fig1.add_trace(
        go.Scatter(
            x=iterations,
            y=all_means,
            mode="lines+markers",
            name=f"{task.capitalize()} Mean",
            line=dict(color="blue"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=[iterations[0], iterations[0]],
            y=[init_mean, init_mean],
            mode="lines",
            name="Initial Mean",
            line=dict(color="red", dash="dash"),
        )
    )
    fig1.update_layout(
        title=f"Full Convergence of {task.capitalize()} Mean",
        xaxis_title="Iterations",
        yaxis_title=f"{task.capitalize()} Mean",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Second figure: Zoomed convergence plot
    fig2.add_trace(
        go.Scatter(
            x=iterations,
            y=all_means,
            mode="lines+markers",
            name=f"{task.capitalize()} Mean",
            line=dict(color="green"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[iterations[0], iterations[0]],
            y=[init_mean, init_mean],
            mode="lines",
            name="Initial Mean",
            line=dict(color="red", dash="dash"),
        )
    )
    fig2.update_layout(
        title=f"Zoomed Convergence (Last 50%)",
        xaxis_title="Iterations",
        yaxis_title=f"{task.capitalize()} Mean",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Third figure: Standard deviation (normal scale)
    fig3.add_trace(
        go.Scatter(
            x=iterations,
            y=all_stds,
            mode="lines+markers",
            name="Standard Deviation",
            line=dict(color="red"),
        )
    )
    fig3.update_layout(
        title="Standard Deviation (Linear Scale)",
        xaxis_title="Iterations",
        yaxis_title="Standard Deviation",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Fourth figure: Standard deviation (log scale)
    fig4.add_trace(
        go.Scatter(
            x=iterations,
            y=all_stds,
            mode="lines+markers",
            name="Standard Deviation",
            line=dict(color="magenta"),
        )
    )
    fig4.update_layout(
        title="Standard Deviation (Log Scale)",
        xaxis_title="Iterations",
        yaxis_title="Standard Deviation (log)",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )
    fig4.update_yaxes(type="log", tickformat=".1e", exponentformat="power")

    # Fifth figure: Active nodes count
    fig5.add_trace(
        go.Scatter(
            x=iterations,
            y=active_nodes_counts,
            mode="lines+markers",
            name="Active Nodes",
            line=dict(color="green"),
        )
    )
    fig5.update_layout(
        title="Active Nodes Over Iterations",
        xaxis_title="Iterations",
        yaxis_title="Number of Nodes",
        title_x=0.5,  # Center the title
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=60),
    )

    # Add grid to all plots
    for fig in [fig1, fig2, fig3, fig4, fig5]:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

    # Create a list of figures
    figures = [fig1, fig2, fig3, fig4, fig5]

    # Initial plot
    yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
        3
    ], figures[4]

    # For convergence tracking
    prev_values = None
    stable_rounds = 0
    converged = False  # Add flag to track if convergence has been reached

    # Run gossip algorithm in smaller batches for smoother updates
    batch_size = min(5, stats_interval)  # Process a small batch at a time
    total_batches = int(np.ceil(num_iterations / batch_size))

    for batch in range(total_batches):
        if converged:  # Check if we've already converged and skip remaining batches
            break

        start_iter = batch * batch_size
        end_iter = min((batch + 1) * batch_size, num_iterations)

        # Update status message
        status = f"Processing iterations {start_iter+1}-{end_iter}/{num_iterations}"

        # Process this batch of iterations
        for iteration in range(start_iter, end_iter):
            # Use the modular function for a single gossip iteration
            success, active_count = perform_gossip_iteration(
                nodes, update_func, mode, dropout_prob, dropout_corr
            )

            if not success:
                logs.append(f"Warning: No active nodes in iteration {iteration + 1}")
                continue

            # Check for convergence if enabled
            if convergence_rounds > 0:
                current_values = [node.value for node in nodes]
                if check_convergence(prev_values, current_values):
                    stable_rounds += 1
                    if stable_rounds >= convergence_rounds:
                        logs.append(
                            f"Converged after {iteration + 1} iterations (stable for {convergence_rounds} rounds)."
                        )
                        # Skip to final statistics
                        status = "Converged - simulation complete"
                        end_iter = iteration + 1
                        converged = True  # Set the convergence flag
                        break
                else:
                    stable_rounds = 0
                prev_values = current_values.copy()

            # Update statistics if needed
            if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
                values = [node.value for node in nodes]
                current_mean = mean_function(values)
                current_std = np.std(values)

                logs.append(
                    f"Stats after {iteration + 1} iterations: Mean = {current_mean:.4f}, "
                    f"Std Dev = {current_std:.4f}, Active Nodes = {active_count}/{num_nodes}"
                )

                # Store for plotting
                all_means.append(current_mean)
                all_stds.append(current_std)
                active_nodes_counts.append(active_count)
                iterations.append(iteration + 1)
                all_values_at_intervals.append(values)

        # Update plot after processing this batch
        if batch % 2 == 0 or batch == total_batches - 1:
            # If we haven't added data for current iteration, add it now for visualization
            if not iterations or iterations[-1] < end_iter:
                values = [node.value for node in nodes]
                current_mean = mean_function(values)
                current_std = np.std(values)
                active_count = sum(1 for node in nodes if node.active)

                all_means.append(current_mean)
                all_stds.append(current_std)
                active_nodes_counts.append(active_count)
                iterations.append(end_iter)
                all_values_at_intervals.append(values)

            # Update each figure separately with the new data
            # Update fig1 (full convergence)
            fig1.data[0].x = iterations
            fig1.data[0].y = all_means
            fig1.data[1].x = [iterations[0], iterations[-1]]
            fig1.data[1].y = [init_mean, init_mean]

            # Update fig2 (zoomed convergence)
            fig2.data[0].x = iterations
            fig2.data[0].y = all_means
            fig2.data[1].x = [iterations[0], iterations[-1]]
            fig2.data[1].y = [init_mean, init_mean]

            # Update fig3 (std dev linear)
            fig3.data[0].x = iterations
            fig3.data[0].y = all_stds

            # Update fig4 (std dev log)
            fig4.data[0].x = iterations
            fig4.data[0].y = all_stds

            # Update fig5 (active nodes)
            fig5.data[0].x = iterations
            fig5.data[0].y = active_nodes_counts

            # Add box plots for data distribution
            if len(all_values_at_intervals) > 1:
                # Clear existing box plots from fig1
                fig1.data = [trace for trace in fig1.data if trace.type != "box"]
                # Make sure we keep our original traces
                if len(fig1.data) < 2:
                    fig1.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=all_means,
                            mode="lines+markers",
                            name=f"{task.capitalize()} Mean",
                            line=dict(color="blue"),
                        )
                    )
                    fig1.add_trace(
                        go.Scatter(
                            x=[iterations[0], iterations[-1]],
                            y=[init_mean, init_mean],
                            mode="lines",
                            name="Initial Mean",
                            line=dict(color="red", dash="dash"),
                        )
                    )

                # Add box plots to fig1 (full convergence)
                for i, values in enumerate(all_values_at_intervals):
                    fig1.add_trace(
                        go.Box(
                            y=values,
                            x=[iterations[i]] * len(values),
                            name=f"Iteration {iterations[i]}",
                            boxpoints=False,
                            marker_color="rgba(0, 0, 255, 0.3)",
                            showlegend=False,
                            line_width=1,
                            width=max(1, num_iterations / 50),
                        )
                    )

                # Clear existing box plots from fig2
                fig2.data = [trace for trace in fig2.data if trace.type != "box"]
                # Make sure we keep our original traces
                if len(fig2.data) < 2:
                    fig2.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=all_means,
                            mode="lines+markers",
                            name=f"{task.capitalize()} Mean",
                            line=dict(color="green"),
                        )
                    )
                    fig2.add_trace(
                        go.Scatter(
                            x=[iterations[0], iterations[-1]],
                            y=[init_mean, init_mean],
                            mode="lines",
                            name="Initial Mean",
                            line=dict(color="red", dash="dash"),
                        )
                    )

                # Only show the last 50% of data points for zoomed plot
                half_point = max(1, len(all_values_at_intervals) // 2)
                if len(all_values_at_intervals) > half_point:
                    zoomed_values = all_values_at_intervals[half_point:]
                    zoomed_iterations = iterations[half_point:]

                    for i, values in enumerate(zoomed_values):
                        fig2.add_trace(
                            go.Box(
                                y=values,
                                x=[zoomed_iterations[i]] * len(values),
                                name=f"Iteration {zoomed_iterations[i]}",
                                boxpoints=False,
                                marker_color="rgba(0, 255, 0, 0.3)",
                                showlegend=False,
                                line_width=1,
                                width=max(1, num_iterations / 50),
                            )
                        )

            # Set axis ranges for better visualization
            # For full convergence plot (fig1)
            y_min = min(min(all_means), init_mean) * 0.95
            y_max = max(max(all_means), init_mean) * 1.05
            fig1.update_yaxes(range=[y_min, y_max])

            # For zoomed plot (fig2)
            if len(all_means) > 2:
                half_index = max(1, len(all_means) // 2)
                zoomed_means = all_means[half_index:]
                if zoomed_means:
                    min_val = min(min(zoomed_means), init_mean) * 0.98
                    max_val = max(max(zoomed_means), init_mean) * 1.02
                    fig2.update_yaxes(range=[min_val, max_val])

                    half_point_x = iterations[half_index]
                    fig2.update_xaxes(range=[half_point_x, iterations[-1]])

            yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
                3
            ], figures[4]

    # Final update with complete stats
    final_values = [node.value for node in nodes]
    final_mean = mean_function(final_values)
    final_std = np.std(final_values)
    active_count = sum(1 for node in nodes if node.active)

    logs.append("\nFinal results:")
    logs.append(f"Final mean: {final_mean:.4f}")
    logs.append(f"Final std dev: {final_std:.4f}")
    logs.append(f"Active nodes: {active_count}/{num_nodes}")

    # Make sure the final iteration is in the plot
    if iterations[-1] != num_iterations:
        all_means.append(final_mean)
        all_stds.append(final_std)
        active_nodes_counts.append(active_count)
        iterations.append(num_iterations)
        all_values_at_intervals.append(final_values)

        # Update plot with final data
        fig1.data[0].x = iterations
        fig1.data[0].y = all_means
        fig1.data[1].x = [iterations[0], iterations[-1]]
        fig1.data[1].y = [init_mean, init_mean]
        fig2.data[0].x = iterations
        fig2.data[0].y = all_means
        fig2.data[1].x = [iterations[0], iterations[-1]]
        fig2.data[1].y = [init_mean, init_mean]
        fig3.data[0].x = iterations
        fig3.data[0].y = all_stds
        fig4.data[0].x = iterations
        fig4.data[0].y = all_stds
        fig5.data[0].x = iterations
        fig5.data[0].y = active_nodes_counts

    status = "Simulation complete"
    yield "\n".join(logs), status, figures[0], figures[1], figures[2], figures[
        3
    ], figures[4]


# Create the Gradio interface
with gr.Blocks(
    title="Gossip Algorithm Simulator",
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
    gr.Markdown("# Gossip Algorithm Simulator")
    gr.Markdown(
        """
    Simulate different gossip algorithms to compute means (geometric, harmonic, and arithmetic) 
    across a distributed network of nodes. The visualization updates smoothly as the simulation progresses.
    """
    )

    # Input controls section
    with gr.Row():
        # Left column for parameters
        with gr.Column(scale=2):
            # Input components
            num_nodes = gr.Slider(
                minimum=10, maximum=10000, value=1000, step=10, label="Number of Nodes"
            )
            num_iterations = gr.Slider(
                minimum=10, maximum=5000, value=50, step=5, label="Number of Iterations"
            )
            task = gr.Radio(
                ["geometric", "harmonic", "arithmetic"],
                value="geometric",
                label="Mean Calculation Type",
            )
            mode = gr.Radio(
                ["push-only", "pull-only", "push-pull"],
                value="push-pull",
                label="Gossip Method",
            )

        # Right column for more parameters
        with gr.Column(scale=2):
            stats_interval = gr.Slider(
                minimum=1, maximum=100, value=5, step=1, label="Statistics Interval"
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
                label="Full Convergence", elem_classes="plot-container"
            )
            output_plot2 = gr.Plot(
                label="Zoomed Convergence", elem_classes="plot-container"
            )
            output_plot3 = gr.Plot(
                label="Standard Deviation (Linear)", elem_classes="plot-container"
            )
            output_plot4 = gr.Plot(
                label="Standard Deviation (Log)", elem_classes="plot-container"
            )
            output_plot5 = gr.Plot(label="Active Nodes", elem_classes="plot-container")

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
            task,
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
        ],
    )

    # Add examples for quick testing
    gr.Examples(
        examples=[
            [1000, 50, "geometric", "push-pull", 5, 42, 0.0, 0.0, 3],
            [1000, 50, "arithmetic", "push-pull", 5, 42, 0.2, 0.5, 0],
            [500, 100, "harmonic", "push-only", 10, 123, 0.1, 0.3, 5],
        ],
        inputs=[
            num_nodes,
            num_iterations,
            task,
            mode,
            stats_interval,
            seed,
            dropout_prob,
            dropout_corr,
            convergence_rounds,
        ],
    )

if __name__ == "__main__":
    # This ensures seed is set even before gradio interface is created
    random.seed(42)  # Default seed
    global_random.seed(42)
    np.random.seed(42)

    # Launch the demo without CSS parameter
    demo.launch()

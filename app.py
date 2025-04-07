import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from statistics import geometric_mean, harmonic_mean, mean

# Import from main.py
from main import Node, initialize_nodes, get_update_func, get_mean_function, perform_gossip_iteration

def run_simulation(
    num_nodes, 
    num_iterations, 
    task, 
    mode, 
    stats_interval, 
    seed, 
    dropout_prob, 
    dropout_corr
):
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
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
    logs = [f"Starting simulation with {num_nodes} nodes...",
            f"Initial mean: {init_mean:.4f}, Initial std dev: {init_std:.4f}"]
    
    status = "Initializing simulation..."
    
    # Create the plotting components with 5 subplots instead of 3
    fig = plt.figure(figsize=(14, 20))
    
    # Define grid for subplots with more space between them
    gs = fig.add_gridspec(5, 1, hspace=0.5)
    
    # First subplot: Full convergence plot (normal scale)
    ax1 = fig.add_subplot(gs[0])
    line_mean_full, = ax1.plot(iterations, all_means, 'b-o', label=f'{task.capitalize()} Mean')
    # Add horizontal dashed line for initial mean
    ax1.axhline(y=init_mean, color='r', linestyle='--', alpha=0.7, label='Initial Mean')
    
    # Second subplot: Zoomed convergence plot (last 50%)
    ax2 = fig.add_subplot(gs[1])
    line_mean_zoomed, = ax2.plot(iterations, all_means, 'g-o', label=f'{task.capitalize()} Mean (Last 50%)')
    # Add horizontal dashed line for initial mean
    ax2.axhline(y=init_mean, color='r', linestyle='--', alpha=0.7, label='Initial Mean')
    
    # Third subplot: Standard deviation (normal scale)
    ax3 = fig.add_subplot(gs[2])
    line_std_normal, = ax3.plot(iterations, all_stds, 'r-o', label='Standard Deviation')
    
    # Fourth subplot: Standard deviation (log scale)
    ax4 = fig.add_subplot(gs[3])
    line_std_log, = ax4.plot(iterations, all_stds, 'm-o', label='Standard Deviation (Log Scale)')
    ax4.set_yscale('log')
    
    # Fifth subplot: Active nodes count
    ax5 = fig.add_subplot(gs[4])
    line_active, = ax5.plot(iterations, active_nodes_counts, 'g-o', label='Active Nodes')
    
    # Configure the plots
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mean Value')
    ax1.set_title(f'Full Convergence of {task.capitalize()} Mean')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Mean Value')
    ax2.set_title(f'Zoomed Convergence (Last 50%)')
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Standard Deviation (Linear Scale)')
    ax3.grid(True)
    ax3.legend()
    
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Standard Deviation (Log Scale)')
    ax4.grid(True)
    ax4.legend()
    
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('Number of Active Nodes')
    ax5.set_title('Active Nodes Over Iterations')
    ax5.grid(True)
    ax5.legend()
    
    # Replace tight_layout() with manual adjustment
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.5)
    
    # Initial plot
    yield "\n".join(logs), status, fig
    
    # Function to update the plot without recreating it
    def update_plot():
        # Update line data for all plots
        line_mean_full.set_xdata(iterations)
        line_mean_full.set_ydata(all_means)
        
        line_mean_zoomed.set_xdata(iterations)
        line_mean_zoomed.set_ydata(all_means)
        
        line_std_normal.set_xdata(iterations)
        line_std_normal.set_ydata(all_stds)
        
        line_std_log.set_xdata(iterations)
        line_std_log.set_ydata(all_stds)
        
        line_active.set_xdata(iterations)
        line_active.set_ydata(active_nodes_counts)
        
        # Clear previous box plots before adding new ones
        for artist in ax1.artists:
            artist.remove()
        for poly in ax1.collections:
            if type(poly).__name__ == 'PolyCollection':  # Box plot elements
                poly.remove()
        
        for artist in ax2.artists:
            artist.remove()
        for poly in ax2.collections:
            if type(poly).__name__ == 'PolyCollection':
                poly.remove()
        
        # Add box plots to full convergence plot
        if len(all_values_at_intervals) > 1:  # Skip if we only have initial values
            # Set positions for box plots (align with iterations)
            positions = iterations
            # Create box plots for full convergence
            box_plot_full = ax1.boxplot(all_values_at_intervals, positions=positions, 
                                     widths=max(1, num_iterations/50),
                                     patch_artist=True, showfliers=False)
            
            # Style the box plots
            for box in box_plot_full['boxes']:
                box.set(color='blue', alpha=0.3)
            
            # For zoomed plot, only show the last 50% of data points
            half_point = max(1, len(all_values_at_intervals) // 2)
            if len(all_values_at_intervals) > half_point:
                zoomed_values = all_values_at_intervals[half_point:]
                zoomed_positions = iterations[half_point:]
                
                # Create box plots for zoomed convergence
                box_plot_zoomed = ax2.boxplot(zoomed_values, positions=zoomed_positions,
                                           widths=max(1, num_iterations/50),
                                           patch_artist=True, showfliers=False)
                
                # Style the box plots
                for box in box_plot_zoomed['boxes']:
                    box.set(color='green', alpha=0.3)
                
                # Set x-axis limits for zoomed plot to show only last 50%
                if len(zoomed_positions) > 0:
                    ax2.set_xlim(left=zoomed_positions[0] - (num_iterations*0.05), 
                               right=iterations[-1] + (num_iterations*0.05))
        
        # Adjust axes limits
        ax1.relim()
        ax1.autoscale_view()
        
        # For zoomed plot, keep the y-axis limits to focus on convergence details
        if len(all_means) > 2:
            half_index = max(1, len(all_means) // 2)
            zoomed_means = all_means[half_index:]
            if zoomed_means:
                min_val = min(zoomed_means) * 0.98
                max_val = max(zoomed_means) * 1.02
                ax2.set_ylim(bottom=min_val, top=max_val)
        
        ax3.relim()
        ax3.autoscale_view()
        ax4.relim()
        ax4.autoscale_view()
        ax5.relim()
        ax5.autoscale_view()
        
        # Replace tight_layout() with manual adjustment
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, hspace=0.5)
    
    # Run gossip algorithm in smaller batches for smoother updates
    batch_size = min(5, stats_interval)  # Process a small batch at a time
    total_batches = int(np.ceil(num_iterations / batch_size))
    
    for batch in range(total_batches):
        start_iter = batch * batch_size
        end_iter = min((batch + 1) * batch_size, num_iterations)
        
        # Update status message
        status = f"Processing iterations {start_iter+1}-{end_iter}/{num_iterations}"
        
        # Process this batch of iterations
        for iteration in range(start_iter, end_iter):
            # Use the modular function for a single gossip iteration
            success, active_count = perform_gossip_iteration(nodes, update_func, mode, dropout_prob, dropout_corr)
            
            if not success:
                logs.append(f"Warning: No active nodes in iteration {iteration + 1}")
                continue
            
            # Update statistics if needed
            if stats_interval > 0 and (iteration + 1) % stats_interval == 0:
                values = [node.value for node in nodes]
                current_mean = mean_function(values)
                current_std = np.std(values)
                
                logs.append(f"Stats after {iteration + 1} iterations: Mean = {current_mean:.4f}, "
                           f"Std Dev = {current_std:.4f}, Active Nodes = {active_count}/{num_nodes}")
                
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
            
            # Update the plot
            update_plot()
            yield "\n".join(logs), status, fig
    
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
        update_plot()
    
    status = "Simulation complete"
    yield "\n".join(logs), status, fig

# Create the Gradio interface
with gr.Blocks(title="Gossip Algorithm Simulator") as demo:
    gr.Markdown("# Gossip Algorithm Simulator")
    gr.Markdown("""
    Simulate different gossip algorithms to compute means (geometric, harmonic, and arithmetic) 
    across a distributed network of nodes. The visualization updates smoothly as the simulation progresses.
    """)
    
    with gr.Row():
        with gr.Column():
            # Input components
            num_nodes = gr.Slider(minimum=10, maximum=10000, value=1000, step=10, label="Number of Nodes")
            num_iterations = gr.Slider(minimum=10, maximum=200, value=50, step=5, label="Number of Iterations")
            task = gr.Radio(["geometric", "harmonic", "arithmetic"], value="geometric", label="Mean Calculation Type")
            mode = gr.Radio(["push-only", "pull-only", "push-pull"], value="push-pull", label="Gossip Method")
            stats_interval = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Statistics Interval")
            seed = gr.Number(value=None, label="Random Seed (optional)", precision=0)
            dropout_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Dropout Probability")
            dropout_corr = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.01, label="Dropout Correlation")
            
            run_button = gr.Button("Run Simulation")
        
        with gr.Column():
            # Output components
            with gr.Column(variant="panel"):
                gr.Markdown("### Simulation Progress")
                output_text = gr.Textbox(label="Results Log", lines=10)
            
            with gr.Column(variant="panel"):
                gr.Markdown("### Visualization")
                # Make the status text more prominent
                status_text = gr.Textbox(label="Current Status", lines=1, max_lines=1)
                output_plot = gr.Plot(label="Convergence Plots")
    
    # Connect the run button to the simulation function but don't pass a progress object
    run_button.click(
        fn=run_simulation, 
        inputs=[num_nodes, num_iterations, task, mode, stats_interval, seed, dropout_prob, dropout_corr], 
        outputs=[output_text, status_text, output_plot]
    )
    
    # Add examples for quick testing
    gr.Examples(
        examples=[
            [1000, 50, "geometric", "push-pull", 5, 42, 0.0, 0.0],
            [1000, 50, "arithmetic", "push-pull", 5, 42, 0.2, 0.5],
            [500, 100, "harmonic", "push-only", 10, 123, 0.1, 0.3],
        ],
        inputs=[num_nodes, num_iterations, task, mode, stats_interval, seed, dropout_prob, dropout_corr],
    )

if __name__ == "__main__":
    demo.launch()

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

def load_history_files(dataset_name):
    """Load all history files for a given dataset"""
    files = glob(f"history/{dataset_name}_*_history.npz")
    histories = {}
    
    for file in files:
        model_name = file.split('_')[-2]  # Extract model name from filename
        data = np.load(file, allow_pickle=True)
        histories[model_name] = {
            'weights': data['weights'],
            'ece': data['ece'],
            'brier': data['brier'],
            'exploration_rate': data['exploration_rate'],
            'augmentation_names': data['augmentation_names']
        }
    return histories

def plot_weight_trajectories(histories, dataset_name, plot_every_n=50):
    """
    Plot weight trajectories with augmentations averaged across models
    including shaded areas for standard deviation
    
    Parameters:
    - histories: Dictionary containing optimization histories
    - dataset_name: Name of the dataset (for title/filename)
    - plot_every_n: Plot every n-th iteration (default=1 means plot all)
    """
    plt.figure(figsize=(12, 8))
    
    # Set font sizes
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # First collect all unique augmentation types
    all_augs = []
    for history in histories.values():
        all_augs.extend(history['augmentation_names'])
    all_augs = sorted(set(all_augs))  # Get unique sorted augmentations
    
    # Custom dark color palette with good separation
    custom_colors = [
        '#1f77b4',  # dark blue
        '#d62728',  # dark red
        '#2ca02c',  # dark green
        '#9467bd',  # dark purple
        '#8c564b',  # dark brown
        '#e377c2',  # dark pink
        '#7f7f7f',  # dark gray
        '#bcbd22',  # dark yellow-green
        '#17becf',  # dark cyan
        '#ff7f0e',  # dark orange
    ]
    
    # Create a color map for augmentations
    aug_colors = {aug: custom_colors[i % len(custom_colors)] for i, aug in enumerate(all_augs)}
    
    # Create a mapping for legend labels
    legend_labels = {
        'gaussian_noise': 'Gaussian Noise',
        'band_stop_filter': 'Band Stop Filter',
        'time_stretch': 'Time Stretch',
        'pitch_shift': 'Pitch Shift',
        'background_noise': 'Background Noise',
        'frequency_mask': 'Frequency Mask',
        'time_mask': 'Time Mask',
        'spec_augment': 'Spec Augment',
        'mixup': 'Mixup',
        'cutmix': 'Cutmix'
    }
    
    # Prepare storage for averaged weights and std dev
    aug_weights = {aug: [] for aug in all_augs}
    aug_stds = {aug: [] for aug in all_augs}  # New: store standard deviations
    iteration_points = []  # To store which iterations we're actually plotting
    max_iterations = max(len(h['weights']) for h in histories.values())
    
    # Average weights across models for each augmentation
    for iteration in range(max_iterations):
        # Skip iterations if not at our plotting interval
        if iteration % plot_every_n != 0 and iteration != max_iterations - 1:
            continue
            
        iteration_points.append(iteration)
        
        # Temporary storage for this iteration
        iter_weights = {aug: [] for aug in all_augs}
        
        for model_name, history in histories.items():
            # If this model has fewer iterations, skip
            if iteration >= len(history['weights']):
                continue
                
            weights = history['weights'][iteration]
            aug_names = history['augmentation_names']
            
            for i, aug in enumerate(aug_names):
                if aug in all_augs:  # Only process known augmentations
                    iter_weights[aug].append(weights[i])
        
        # Compute average and std dev for this iteration
        for aug in all_augs:
            if iter_weights[aug]:  # Only if we have data for this aug
                aug_weights[aug].append(np.mean(iter_weights[aug]))
                aug_stds[aug].append(np.std(iter_weights[aug]))  # New: calculate std
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels_list = []
    
    # Plot each augmentation's average trajectory with shaded area
    for aug in all_augs:
        if aug_weights[aug]:  # Only plot if we have data
            weights_array = np.array(aug_weights[aug])
            std_array = np.array(aug_stds[aug])
            
            # Plot the mean line
            line = plt.plot(iteration_points, weights_array, 
                    color=aug_colors[aug],
                    linewidth=2,
                    marker='o', markersize=4)[0]
            
            # Add shaded area for standard deviation
            plt.fill_between(iteration_points,
                           weights_array - std_array,
                           weights_array + std_array,
                           color=aug_colors[aug],
                           alpha=0.2)
            
            # Create a patch for the legend
            patch = plt.Rectangle((0, 0), 1, 1, fc=aug_colors[aug])
            legend_handles.append(patch)
            legend_labels_list.append(legend_labels.get(aug, aug))
    
    # Add horizontal line at initial weight value
    initial_weight = 1.0/len(all_augs)
    plt.axhline(y=initial_weight, color='gray', linestyle='--', alpha=0.3)
    
    # Adjust x-axis ticks to show clean iteration numbers
    if plot_every_n > 1:
        plt.xticks(iteration_points)
    
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Average Weight Value', fontsize=20)
    
    # Create legend with patches
    plt.legend(legend_handles, legend_labels_list,
              bbox_to_anchor=(0.5, 1.15), loc='upper center', 
              ncol=3, frameon=False, fontsize=18)
    
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.subplots_adjust(top=0.85)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{dataset_name}_avg_weight_trajectories_with_std.png', 
               bbox_inches='tight', dpi=300)
    plt.close()

def plot_metric_progress(histories, dataset_name):
    """Plot ECE and Brier score progress for all models"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Set font sizes
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    for model_name, history in histories.items():
        iterations = range(len(history['ece']))
        
        # Plot ECE
        ax1.plot(iterations, history['ece'], 
                label=f"{model_name}", marker='o', markersize=3)
        
        # Plot Brier
        ax2.plot(iterations, history['brier'],
                label=f"{model_name}", marker='o', markersize=3)
    
    ax1.set_ylabel('ECE', fontsize=14)
    ax1.set_title(f'Calibration Metric Progress\nDataset: {dataset_name}', fontsize=14)
    ax1.grid(True)
    ax1.legend(fontsize=12)
    
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('Brier Score', fontsize=14)
    ax2.grid(True)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{dataset_name}_metric_progress.png', 
               bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., affia3k, mrsffia)')
    args = parser.parse_args()
    
    # Load all histories for this dataset
    histories = load_history_files(args.dataset)
    
    # Generate plots
    plot_weight_trajectories(histories, args.dataset)
    # plot_metric_progress(histories, args.dataset)
    
    print(f"Visualizations saved to 'figures' directory")
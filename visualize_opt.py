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
    
    # First collect all unique augmentation types
    all_augs = []
    for history in histories.values():
        all_augs.extend(history['augmentation_names'])
    all_augs = sorted(set(all_augs))  # Get unique sorted augmentations
    
    # Create a color map for augmentations
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_augs)))
    aug_colors = {aug: colors[i] for i, aug in enumerate(all_augs)}
    
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
    
    # Plot each augmentation's average trajectory with shaded area
    for aug in all_augs:
        if aug_weights[aug]:  # Only plot if we have data
            weights_array = np.array(aug_weights[aug])
            std_array = np.array(aug_stds[aug])
            
            # Plot the mean line
            plt.plot(iteration_points, weights_array, 
                    label=aug, 
                    color=aug_colors[aug],
                    linewidth=2,
                    marker='o', markersize=4)
            
            # Add shaded area for standard deviation
            plt.fill_between(iteration_points,
                           weights_array - std_array,
                           weights_array + std_array,
                           color=aug_colors[aug],
                           alpha=0.2)
    
    # Add horizontal line at initial weight value
    initial_weight = 1.0/len(all_augs)
    plt.axhline(y=initial_weight, color='gray', linestyle='--', alpha=0.3)
    
    # Adjust x-axis ticks to show clean iteration numbers
    if plot_every_n > 1:
        plt.xticks(iteration_points)
    
    # plt.title(f'Average Ensemble Weights with Standard Deviation\n{dataset_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Weight Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{dataset_name}_avg_weight_trajectories_with_std.png', 
               bbox_inches='tight', dpi=300)
    plt.close()

def plot_metric_progress(histories, dataset_name):
    """Plot ECE and Brier score progress for all models"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for model_name, history in histories.items():
        iterations = range(len(history['ece']))
        
        # Plot ECE
        ax1.plot(iterations, history['ece'], 
                label=f"{model_name}", marker='o', markersize=3)
        
        # Plot Brier
        ax2.plot(iterations, history['brier'],
                label=f"{model_name}", marker='o', markersize=3)
    
    ax1.set_ylabel('ECE')
    ax1.set_title(f'Calibration Metric Progress\nDataset: {dataset_name}')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Brier Score')
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
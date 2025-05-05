import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
import re
from datetime import datetime


def cumulative_max_per_round(sequences):
    num_rounds = sequences['round'].max() + 1
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]

    return np.maximum.accumulate(max_per_round)


def get_all_csv_files(directory, pattern):
    """Get all CSV files matching the pattern in the directory"""
    path_pattern = os.path.join(directory, f"*{pattern}*.csv")
    csv_files = glob.glob(path_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory} matching pattern {pattern}")
    
    return csv_files


def calculate_stats_across_runs(directory, pattern, truncated=None):
    """Calculate mean and standard deviation across multiple runs"""
    csv_files = get_all_csv_files(directory, pattern)
    print(f"Found {len(csv_files)} files matching pattern {pattern} in {directory}")
    
    all_runs_data = []
    
    # Process each CSV file
    for file_path in csv_files:
        with open(file_path) as f:
            try:
                metadata = json.loads(next(f))  # Skip metadata line
                data = pd.read_csv(f)
                max_per_round = cumulative_max_per_round(data)
                all_runs_data.append(max_per_round)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy array for easier stats calculation
    # Make sure all runs have the same number of rounds
    min_length = min(len(run) for run in all_runs_data)
    if truncated is not None:
        min_length = min(min_length, truncated)
    all_runs_array = np.array([run[:min_length] for run in all_runs_data])
    
    # Calculate mean and std
    mean_values = np.mean(all_runs_array, axis=0)
    std_values = np.std(all_runs_array, axis=0)
    
    return mean_values, std_values, min_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Combine both CNN and non-CNN landscapes
    all_landscapes = ['aav', 'rna', 'gfp', 'tf', 'rosetta', 'muformer']
    
    # Create a single figure with subplots arranged in 2 columns
    n_landscapes = len(all_landscapes)
    n_rows = (n_landscapes + 1) // 2  # Calculate number of rows needed (rounded up)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows), dpi=300)
    
    # Make axes a 2D array even if we have only one row
    if n_rows == 1:
        axes = np.array([axes])
        
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Dictionary to store all lines for the shared legend
    legend_handles = []
    legend_labels = []
    
    # Process each landscape
    for i, landscape in enumerate(all_landscapes):
        ax = axes[i]
        
        for sequences_batch_size, model_queries_per_batch in [(100, 5000)]:
            ax.set_title(f'Performance on {landscape}', fontweight='bold')
            
            # Choose appropriate method list based on landscape
            if landscape in ['aav', 'rna', 'gfp', 'tf', 'rosetta']:
                method_list = ['adalead', 'cmaes', 'dynappo', 'musearch', 'BO', 'cbas', 'evoplay', 'pex', 'pure_random', 'gwg']
                dirichlet_ppo_list = ['musearch']
                is_cnn = True
            else:
                method_list = ['pure_random', 'adalead_gt', 'BO_gt', 'musearch_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'gwg_gt']
                dirichlet_ppo_list = ['musearch_gt']
                is_cnn = False
            
            # Define method name mapping for display
            method_display_names = {
                'musearch': 'μSearch', 'musearch_gt': 'μSearch',
                'adalead': 'AdaLead', 'adalead_gt': 'AdaLead',
                'cmaes': 'CMA-ES', 'cmaes_gt': 'CMA-ES',
                'BO': 'BO', 'BO_gt': 'BO',
                'cbas': 'CbAS', 'cbas_gt': 'CbAS',
                'dynappo': 'DyNA-PPO', 'dynappo_gt': 'DyNA-PPO',
                'pex': 'PEX',
                'evoplay': 'EvoPlay', 'evoplay_gt': 'EvoPlay',
                'gwg': 'GWG', 'gwg_gt': 'GWG',
                'pure_random': 'Random'
            }
                
            for method in method_list:
                pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
                directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
                
                try:
                    mean_values, std_values, num_rounds = calculate_stats_across_runs(directory, pattern, truncated=11)
                    rounds = np.arange(num_rounds) * 5000
                    
                    # Plot mean line with display name
                    display_name = method_display_names.get(method, method)  # Use display name or original if not found
                    line, = ax.plot(rounds, mean_values, '-o', markersize=3, label=display_name)
                    
                    # Add standard deviation as shaded region
                    ax.fill_between(rounds, 
                                  mean_values - std_values, 
                                  mean_values + std_values, 
                                  alpha=0.3)
                    
                    # Only add to legend for the first subplot to avoid duplicates
                    if i == 0:
                        legend_handles.append(line)
                        legend_labels.append(display_name)
                        
                except Exception as e:
                    print(f"Error with {method} on {landscape}: {e}")
            
            ax.set_ylabel("Cumulative max", fontweight='bold')
            ax.set_xlabel("Number of Model Queries", fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    # Create a shared legend outside the subplots
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.98), 
        ncol=5
    )
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the combined figure
    fig.savefig(f'examples/figs/combined_landscapes_comparison.png')
    
    print(f"Combined figure saved to examples/figs/combined_landscapes_comparison.png")
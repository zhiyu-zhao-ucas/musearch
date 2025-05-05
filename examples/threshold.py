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
    parser.add_argument('--cnn', action='store_true')
    args = parser.parse_args()
    
    if args.cnn:
        landscapes = ['aav', 'rna', 'gfp', 'tf', 'rosetta']
    else:
        landscapes = ['muformer']
        
    for landscape in landscapes:
        for sequences_batch_size, model_queries_per_batch in [(100, 5000)]:
            plt.figure(dpi=300, figsize=(10, 8))
            # Add top margin to make room for legend
            plt.subplots_adjust(top=0.85)
            plt.title(f'Performance on {landscape}', fontweight='bold')
            
            if args.cnn:
                method_list = ['adalead', 'cmaes', 'dynappo', 'BO', 'dirichlet_ppo_cnn']
                dirichlet_ppo_list = ['dirichlet_ppo_cnn']
            else:
                method_list = ['musearch_gt', 'pure_random', 'adalead_gt', 'BO_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'cbas_gt']
                dirichlet_ppo_list = ['musearch_gt']
                
            for method in method_list:
                pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
                directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
                
                try:
                    mean_values, std_values, num_rounds = calculate_stats_across_runs(directory, pattern)
                    rounds = np.arange(num_rounds) * 5000
                    
                    # Plot mean line
                    plt.plot(rounds, mean_values, '-o', markersize=3, label=f'{method}')
                    
                    # Add standard deviation as shaded region if requested
                    plt.fill_between(rounds, 
                                        mean_values - std_values, 
                                        mean_values + std_values, 
                                        alpha=0.3)
                        
                except Exception as e:
                    print(f"Error with {method}: {e}")
                
            plt.ylabel("Cumulative max (avg across runs)", fontweight='bold')
            plt.xlabel("Number of Model Queries", fontweight='bold')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4)
            
            # Add grid to the plot
            plt.grid(True, linestyle=':', alpha=0.7)
            
            # Set tick labels to normal weight (not bold)
            plt.tick_params(axis='both', which='major', labelsize=10)
            
            # Save figure
            suffix = "_with_std"
            if args.cnn:
                plt.savefig(f'examples/figs/cnn/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_cnn_avg{suffix}.png')
            else:
                plt.savefig(f'examples/figs/gt/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_gt_avg{suffix}.png')
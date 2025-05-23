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


def plot_landscape(ax, landscape, landscape_names, method_list, color_mapping, method_display_names, methods_in_legend, legend_handles, legend_labels):
    """Plot a single landscape on the provided axis"""
    
    # Letter label is now added in the main function, so we don't add it here anymore
    
    for sequences_batch_size, model_queries_per_batch in [(100, 5000)]:
        ax.set_title(f'Performance on {landscape_names[landscape]}', fontweight='bold')
        
        # Choose appropriate method list based on landscape
        if landscape in ['aav', 'rna', 'gfp', 'tf', 'rosetta']:
            current_method_list = ['adalead', 'BO', 'cmaes',  'cbas', 'musearch', 'dynappo', 'evoplay', 'pex', 'pure_random', 'gwg']
            dirichlet_ppo_list = ['musearch']
            is_cnn = True
        else:
            current_method_list = ['pure_random', 'adalead_gt', 'BO_gt', 'musearch_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'gwg_gt', 'pex_gt']
            dirichlet_ppo_list = ['musearch_gt']
            is_cnn = False
            
        for method in current_method_list:
            pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
            directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
            
            try:
                # Use 50 rounds for muformer, 11 for others
                truncated_rounds = 60 if landscape == 'muformer' else 11
                mean_values, std_values, num_rounds = calculate_stats_across_runs(directory, pattern, truncated=truncated_rounds)
                
                # Set x-axis values based on landscape type
                if landscape == 'muformer':
                    rounds = np.arange(num_rounds) * 5000
                else:
                    rounds = np.arange(num_rounds)
                
                # Get display name
                display_name = method_display_names.get(method, method)
                
                # Use consistent color for this method
                color = color_mapping.get(method, None)
                
                # Plot mean line with consistent color
                line, = ax.plot(rounds, mean_values, '-o', markersize=3, 
                               label=display_name, color=color)
                
                # Add standard deviation as shaded region with same color
                ax.fill_between(rounds, 
                              mean_values - std_values, 
                              mean_values + std_values, 
                              alpha=0.3, color=color)
                
                # Add to legend only if not already added
                base_method = method.replace('_gt', '')
                if base_method not in methods_in_legend:
                    legend_handles.append(line)
                    legend_labels.append(display_name)
                    methods_in_legend.add(base_method)
                    
            except Exception as e:
                print(f"Error with {method} on {landscape}: {e}")
        
        ax.set_ylabel("Maximum Fitness", fontweight='bold')
        
        # Set x-axis label based on landscape type
        if landscape == 'muformer':
            ax.set_xlabel("Number of Model Queries", fontweight='bold')
        else:
            ax.set_xlabel("Number of Rounds", fontweight='bold')
        
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    return methods_in_legend, legend_handles, legend_labels


def save_individual_subplot(landscape, landscape_names, method_list, color_mapping, method_display_names, letter_idx):
    """Save an individual subplot for a landscape"""
    individual_fig = plt.figure(figsize=(6, 5), dpi=300)
    individual_ax = individual_fig.add_subplot(111)
    
    # Add bold letter label to top-left corner
    letter = chr(97 + letter_idx)  # Convert 0 to 'a', 1 to 'b', etc.
    individual_ax.text(-0.1, 1.1, letter, transform=individual_ax.transAxes, 
                       fontsize=14, fontweight='bold', va='top')
    
    for sequences_batch_size, model_queries_per_batch in [(100, 5000)]:
        individual_ax.set_title(f'Performance on {landscape_names[landscape]}', fontweight='bold')
        
        # Choose appropriate method list based on landscape
        if landscape in ['aav', 'rna', 'gfp', 'tf', 'rosetta']:
            current_method_list = ['adalead', 'BO', 'cmaes',  'cbas', 'musearch', 'dynappo', 'evoplay', 'pex', 'pure_random', 'gwg']
        else:
            current_method_list = ['pure_random', 'adalead_gt', 'BO_gt', 'musearch_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'gwg_gt', 'pex_gt']
            
        for method in current_method_list:
            try:
                pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
                directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
                
                # Use 50 rounds for muformer, 11 for others
                truncated_rounds = 51 if landscape == 'muformer' else 11
                mean_values, std_values, num_rounds = calculate_stats_across_runs(directory, pattern, truncated=truncated_rounds)
                
                # Set x-axis values based on landscape type
                if landscape == 'muformer':
                    rounds = np.arange(num_rounds) * 5000
                else:
                    rounds = np.arange(num_rounds)
                
                # Get display name
                display_name = method_display_names.get(method, method)
                
                # Use consistent color for this method
                color = color_mapping.get(method, None)
                
                # Plot line on individual plot
                individual_ax.plot(rounds, mean_values, '-o', markersize=3, 
                                 label=display_name, color=color)
                
                # Add standard deviation as shaded region with same color for individual plot
                individual_ax.fill_between(rounds, 
                              mean_values - std_values, 
                              mean_values + std_values, 
                              alpha=0.3, color=color)
            except Exception as e:
                print(f"Error with {method} on individual plot for {landscape}: {e}")
    
    # Set title and labels
    individual_ax.set_title(f'Performance on {landscape_names[landscape]}', fontweight='bold')
    individual_ax.set_ylabel("Maximum Fitness", fontweight='bold')
    if landscape == 'muformer':
        individual_ax.set_xlabel("Number of Model Queries", fontweight='bold')
    else:
        individual_ax.set_xlabel("Number of Rounds", fontweight='bold')
        
    individual_ax.grid(True, linestyle=':', alpha=0.7)
    individual_ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend to individual plot
    individual_ax.legend(
        [plt.Line2D([0], [0], color=color_mapping.get(method, None)) 
         for method in current_method_list],
        [method_display_names.get(method, method) 
         for method in current_method_list],
        loc='best'
    )
    
    # Adjust layout
    individual_fig.tight_layout()
    
    # Save the individual figure
    output_path = f'examples/figs/{landscape}_comparison.pdf'
    individual_fig.savefig(output_path)
    print(f"Individual figure saved to {output_path}")
    
    plt.close(individual_fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs('examples/figs', exist_ok=True)
    
    # All landscapes
    all_landscapes = ['gfp', 'rosetta', 'aav', 'rna', 'tf', 'muformer']
    
    # Separate landscapes into two groups
    rna_tf_landscapes = ['rna', 'tf']
    other_landscapes = ['gfp', 'rosetta', 'aav', 'muformer']
    
    landscape_names = {
        'gfp': 'GFP',
        'rosetta': 'Rosetta',
        'aav': 'AAV',
        'rna': 'RNA',
        'tf': 'TF-Binding',
        'muformer': 'μFormer based TEM-1'
    }
    
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
    
    # Define consistent colors for each algorithm
    color_mapping = {
        'musearch': '#d62728', 'musearch_gt': '#d62728',  # red
        'adalead': '#1f77b4', 'adalead_gt': '#1f77b4',    # blue
        'cmaes': '#ff7f0e', 'cmaes_gt': '#ff7f0e',        # orange
        'BO': '#2ca02c', 'BO_gt': '#2ca02c',              # green
        'cbas': '#7f7f7f', 'cbas_gt': '#7f7f7f',          # gray
        'dynappo': '#9467bd', 'dynappo_gt': '#9467bd',    # purple
        'pex': '#8c564b', 'pex_gt': '#8c564b',            # brown
        'evoplay': '#bcbd22', 'evoplay_gt': '#bcbd22',    # olive
        'gwg': '#e377c2', 'gwg_gt': '#e377c2',            # pink
        'pure_random': '#17becf'                          # cyan
    }
    
    # Assign sequential letters to all landscapes
    # Use a sequential lettering scheme for all plots
    letter_mapping = {
        'rna': 0,     # a
        'tf': 1,      # b
        'gfp': 2,     # c
        'rosetta': 3, # d
        'aav': 4,     # e
        'muformer': 5 # f
    }
    
    # Save individual plots for each landscape
    for landscape in all_landscapes:
        save_individual_subplot(
            landscape, 
            landscape_names, 
            ['adalead', 'BO', 'cmaes', 'cbas', 'musearch', 'dynappo', 'evoplay', 'pex', 'pure_random', 'gwg'],
            color_mapping, 
            method_display_names,
            letter_mapping[landscape]
        )
    
    # Create figure for RNA and TF
    rna_tf_fig, rna_tf_axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    # Flatten the axes array for easier iteration
    rna_tf_axes = rna_tf_axes.flatten()
    
    # Dictionary to store all lines for the shared legend
    rna_tf_legend_handles = []
    rna_tf_legend_labels = []
    rna_tf_methods_in_legend = set()
    
    # Plot RNA and TF in a 1x2 layout
    for i, landscape in enumerate(rna_tf_landscapes):
        # Manually add letter label to ensure consistent lettering across all plots
        ax = rna_tf_axes[i]
        letter = chr(97 + letter_mapping[landscape])  # Use consistent letter mapping
        ax.text(-0.1, 1.1, letter, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        rna_tf_methods_in_legend, rna_tf_legend_handles, rna_tf_legend_labels = plot_landscape(
            ax, landscape, landscape_names, 
            ['adalead', 'BO', 'cmaes', 'cbas', 'musearch', 'dynappo', 'evoplay', 'pex', 'pure_random', 'gwg'], 
            color_mapping, method_display_names, 
            rna_tf_methods_in_legend, rna_tf_legend_handles, rna_tf_legend_labels)
    
    # Create a shared legend outside the subplots for RNA-TF
    rna_tf_fig.legend(
        rna_tf_legend_handles, 
        rna_tf_legend_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.0), 
        ncol=5
    )
    
    # Adjust layout to make room for the legend - provide just enough space
    rna_tf_fig.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save the RNA-TF figure
    rna_tf_fig.savefig(f'examples/figs/rna_tf_comparison.pdf')
    rna_tf_fig.savefig(f'examples/figs/rna_tf_comparison.png')
    
    print(f"RNA-TF figure saved to examples/figs/rna_tf_comparison.pdf")
    
    # Create figure for other landscapes
    other_fig, other_axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    # Flatten the axes array for easier iteration
    other_axes = other_axes.flatten()
    
    # Dictionary to store all lines for the shared legend
    other_legend_handles = []
    other_legend_labels = []
    other_methods_in_legend = set()
    
    # Plot other landscapes in a 2x2 layout
    for i, landscape in enumerate(other_landscapes):
        # Manually add letter label with 'a' starting for this figure
        ax = other_axes[i]
        letter = chr(97 + i)  # a, b, c, d for this figure
        ax.text(-0.1, 1.1, letter, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top')
        
        other_methods_in_legend, other_legend_handles, other_legend_labels = plot_landscape(
            ax, landscape, landscape_names, 
            ['adalead', 'BO', 'cmaes', 'cbas', 'musearch', 'dynappo', 'evoplay', 'pex', 'pure_random', 'gwg'], 
            color_mapping, method_display_names, 
            other_methods_in_legend, other_legend_handles, other_legend_labels)
    
    # Create a shared legend outside the subplots for other landscapes
    other_fig.legend(
        other_legend_handles, 
        other_legend_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.0), 
        ncol=5
    )
    
    # Adjust layout to make room for the legend
    other_fig.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save the other landscapes figure
    other_fig.savefig(f'examples/figs/other_landscapes_comparison.pdf')
    other_fig.savefig(f'examples/figs/other_landscapes_comparison.png')
    
    print(f"Other landscapes figure saved to examples/figs/other_landscapes_comparison.pdf")
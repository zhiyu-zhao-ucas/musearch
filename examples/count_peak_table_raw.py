import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
import re
from collections import defaultdict

def get_all_csv_files(directory, pattern):
    """Get all CSV files matching the pattern in the directory"""
    path_pattern = os.path.join(directory, f"*{pattern}*.csv")
    csv_files = glob.glob(path_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {directory} matching pattern {pattern}")
        return []
    
    return csv_files

def find_max_min_fitness_for_landscape(landscape, methods, sequences_batch_size=100, model_queries_per_batch=5000, max_round=10):
    """Find the maximum and minimum fitness score across all algorithms for a given landscape, up to max_round"""
    max_fitness = float('-inf')
    min_fitness = float('inf')
    
    for method in methods:
        directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
        pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
        
        csv_files = get_all_csv_files(directory, pattern)
        
        for file_path in csv_files:
            try:
                with open(file_path) as f:
                    metadata = json.loads(next(f))  # Skip metadata line
                    data = pd.read_csv(f)
                    # Only consider data up to max_round
                    filtered_data = data[data['round'] <= max_round]
                    
                    if not filtered_data.empty:
                        current_max = filtered_data['true_score'].max()
                        current_min = filtered_data['true_score'].min()
                        max_fitness = max(max_fitness, current_max)
                        min_fitness = min(min_fitness, current_min)
                        print(f"File: {file_path}, Max score: {current_max}, Min score: {current_min} (up to round {max_round})")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Handle edge cases
    if max_fitness == float('-inf'):
        max_fitness = 0
    if min_fitness == float('inf'):
        min_fitness = 0
    
    return max_fitness, min_fitness

def count_peaks_above_threshold(landscape, methods, absolute_threshold, sequences_batch_size=100, model_queries_per_batch=5000, max_round=10):
    """
    Count how many data points exceed the absolute threshold.
    The threshold is applied directly to the raw score.
    Returns separate counts for each run file (_1, _2, _3).
    """
    results = {}
    
    for method in methods:
        # Create separate dictionaries for each run
        method_data_run1 = defaultdict(list)
        method_data_run2 = defaultdict(list)
        method_data_run3 = defaultdict(list)
        
        directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
        pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
        
        csv_files = get_all_csv_files(directory, pattern)
        
        for file_path in csv_files:
            try:
                # Determine which run this file belongs to (_1, _2, or _3)
                run_match = re.search(r'_(\d+)\.csv$', file_path)
                if not run_match:
                    print(f"Warning: Could not determine run number for {file_path}, skipping")
                    continue
                
                run_number = int(run_match.group(1))
                if run_number not in [1, 2, 3]:
                    print(f"Warning: Unexpected run number {run_number} in {file_path}, skipping")
                    continue
                
                with open(file_path) as f:
                    metadata = json.loads(next(f))  # Skip metadata line
                    data = pd.read_csv(f)
                    
                    # Count data points over threshold in each round
                    num_rounds = min(data['round'].max() + 1, max_round + 1)
                    
                    for r in range(num_rounds):
                        round_data = data[data['round'] <= r]  # Consider only data up to round r
                        
                        # Count points exceeding absolute threshold
                        points_over_threshold = sum(round_data['true_score'] >= absolute_threshold)
                        
                        # Add to the appropriate run's data
                        if run_number == 1:
                            method_data_run1[r].append(points_over_threshold)
                        elif run_number == 2:
                            method_data_run2[r].append(points_over_threshold)
                        elif run_number == 3:
                            method_data_run3[r].append(points_over_threshold)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Calculate the total count for each round across all runs
        peaks_per_round_run1 = []
        peaks_per_round_run2 = []
        peaks_per_round_run3 = []
        
        for r in sorted(method_data_run1.keys()):
            if r <= max_round:  # Only include rounds up to max_round
                count = sum(method_data_run1[r])
                peaks_per_round_run1.append(count)
        
        for r in sorted(method_data_run2.keys()):
            if r <= max_round:  # Only include rounds up to max_round
                count = sum(method_data_run2[r])
                peaks_per_round_run2.append(count)
                
        for r in sorted(method_data_run3.keys()):
            if r <= max_round:  # Only include rounds up to max_round
                count = sum(method_data_run3[r])
                peaks_per_round_run3.append(count)
        
        # Store results for all runs separately
        results[method] = {
            'run1': peaks_per_round_run1,
            'run2': peaks_per_round_run2,
            'run3': peaks_per_round_run3
        }
    
    return results

def analyze_landscape(landscape, absolute_threshold, sequences_batch_size=100, model_queries_per_batch=5000, max_round=10):
    """Analyze a single landscape for peak counts"""
    print(f"Analyzing landscape: {landscape}")
    
    # Method list for muformer (ground truth landscape)
    method_list = ['pure_random', 'adalead_gt', 'BO_gt', 'musearch_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'gwg_gt', 'pex_gt']
    
    # Define method name mapping for display
    method_display_names = {
        'musearch_gt': 'μSearch',
        'adalead_gt': 'AdaLead',
        'cmaes_gt': 'CMA-ES',
        'BO_gt': 'BO',
        'cbas_gt': 'CbAS',
        'dynappo_gt': 'DyNA-PPO',
        'evoplay_gt': 'EvoPlay',
        'gwg_gt': 'GWG',
        'pure_random': 'Random',
        'pex_gt': 'PEX'
    }
    
    # Find max and min fitness for this landscape
    max_fitness, min_fitness = find_max_min_fitness_for_landscape(landscape, method_list, sequences_batch_size, model_queries_per_batch, max_round)
    print(f"Fitness range for {landscape} up to round {max_round}: min={min_fitness}, max={max_fitness}")
    
    # Count peaks for each method
    peak_counts = count_peaks_above_threshold(landscape, method_list, absolute_threshold,
                                              sequences_batch_size, model_queries_per_batch, max_round)
    
    # Print and plot results
    plt.figure(figsize=(10, 6), dpi=300)
    
    for method, runs_data in peak_counts.items():
        display_name = method_display_names.get(method, method)
        
        # Get total number of data points per round (for percentage calculation)
        directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
        pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
        csv_files = get_all_csv_files(directory, pattern)
        
        # Plot aggregated results (all runs combined)
        all_runs_combined = []
        max_len = max(len(runs_data['run1']), len(runs_data['run2']), len(runs_data['run3']))
        
        for i in range(max_len):
            total = 0
            if i < len(runs_data['run1']):
                total += runs_data['run1'][i]
            if i < len(runs_data['run2']):
                total += runs_data['run2'][i]
            if i < len(runs_data['run3']):
                total += runs_data['run3'][i]
            all_runs_combined.append(total)
        
        if all_runs_combined:
            print(f"{display_name}: {all_runs_combined} data points exceed absolute threshold of {absolute_threshold}")
            
            # Plot results
            rounds = np.arange(len(all_runs_combined)) * model_queries_per_batch
            plt.plot(rounds, all_runs_combined, '-o', label=display_name, markersize=3)
    
    plt.title(f'Number of data points exceeding absolute threshold of {absolute_threshold} on {landscape}', fontweight='bold')
    plt.xlabel('Number of Model Queries', fontweight='bold')
    plt.ylabel('Number of Data Points', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    
    # Save figure
    os.makedirs('examples/figs/peak_counts', exist_ok=True)
    plt.savefig(f'examples/figs/peak_counts/absolute_points_over_threshold_{landscape}_{absolute_threshold}_round_{max_round}.png')
    print(f"Analysis complete. Results saved to examples/figs/peak_counts/absolute_points_over_threshold_{landscape}_{absolute_threshold}_round_{max_round}.png")
    print(f"Absolute threshold: {absolute_threshold}")
    
    return max_fitness, min_fitness, peak_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Absolute threshold value to count as a peak')
    parser.add_argument('--max_round', type=int, default=50,
                        help='Maximum round to consider (default: 30)')
    args = parser.parse_args()
    
    # Only analyze muformer landscape
    landscape = 'muformer'
    
    # Analyze the landscape
    max_fitness, min_fitness, peak_counts = analyze_landscape(landscape, args.threshold, max_round=args.max_round)
    
    # Save results to text file with combined data
    results_filename = f'examples/figs/peak_counts/results_{landscape}_{args.threshold}_round_{args.max_round}.md'
    with open(results_filename, 'w') as f:
        f.write(f"# Analysis for landscape: {landscape}\n\n")
        f.write(f"- **Absolute threshold**: {args.threshold}\n")
        f.write(f"- **Max round**: {args.max_round}\n")
        f.write(f"- **Fitness range**: min={min_fitness}, max={max_fitness}\n\n")
        print(f"threshold ratio: {(args.threshold - min_fitness) / (max_fitness - min_fitness)*100}%")
        f.write(f"- **threshold ratio**: {(args.threshold - min_fitness) / (max_fitness - min_fitness)*100} \%\n\n")
        
        f.write("## Data points exceeding threshold by method and round (Combined)\n\n")
        
        # Define method name mapping for display
        method_display_names = {
            'musearch_gt': 'μSearch',
            'adalead_gt': 'AdaLead',
            'cmaes_gt': 'CMA-ES',
            'BO_gt': 'BO',
            'cbas_gt': 'CbAS',
            'dynappo_gt': 'DyNA-PPO',
            'evoplay_gt': 'EvoPlay',
            'gwg_gt': 'GWG',
            'pure_random': 'Random',
            'pex_gt': 'PEX'
        }
        
        # Create a table header with rounds
        max_rounds = 0
        for method, runs_data in peak_counts.items():
            run1_len = len(runs_data['run1'])
            run2_len = len(runs_data['run2'])
            run3_len = len(runs_data['run3'])
            max_rounds = max(max_rounds, run1_len, run2_len, run3_len)
        
        rounds = [r for r in range(max_rounds)]
        header = "| Method "
        for r in rounds:
            header += f"| Round {r} "
        header += "|"
        f.write(f"{header}\n")
        
        # Write markdown table separator
        separator = "|" + "-" * 13 + "|"
        for _ in rounds:
            separator += "-" * 18 + "|"
        f.write(f"{separator}\n")
        
        # Write combined data for each method
        for method, runs_data in peak_counts.items():
            display_name = method_display_names.get(method, method)
            line = f"| {display_name:<11}"
            
            run1 = runs_data['run1']
            run2 = runs_data['run2']
            run3 = runs_data['run3']
            
            for r in rounds:
                total = 0
                if r < len(run1):
                    total += run1[r]
                if r < len(run2):
                    total += run2[r]
                if r < len(run3):
                    total += run3[r]
                line += f"| {total:<16}"
            
            line += "|"
            f.write(f"{line}\n")
    
    print(f"Combined results saved to {results_filename}")
    
    # Save a new markdown file with only the last round data for each run separately
    last_round_filename = f'examples/figs/peak_counts/last_round_{landscape}_{args.threshold}_round_{args.max_round}.md'
    with open(last_round_filename, 'w') as f:
        f.write(f"# Last Round Analysis for landscape: {landscape}\n\n")
        f.write(f"- **Absolute threshold**: {args.threshold}\n")
        f.write(f"- **Max round**: {args.max_round}\n")
        f.write(f"- **Fitness range**: min={min_fitness}, max={max_fitness}\n\n")
        print(f"threshold ratio: {(args.threshold - min_fitness) / (max_fitness - min_fitness)*100}%")
        f.write(f"- **threshold ratio**: {(args.threshold - min_fitness) / (max_fitness - min_fitness)*100} \%\n\n")
        
        f.write("## Data points exceeding threshold at final round (Separated by run)\n\n")
        
        # Create a table with method and last round data for each run
        header = "| Method | Run 1 Final Round | Run 2 Final Round | Run 3 Final Round | Combined Total |"
        f.write(f"{header}\n")
        
        # Write markdown table separator
        separator = "|" + "-" * 13 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 16 + "|"
        f.write(f"{separator}\n")
        
        # Write the last column data for each method and run
        for method, runs_data in peak_counts.items():
            display_name = method_display_names.get(method, method)
            
            # Get last count for each run
            last_count_run1 = runs_data['run1'][-1] if runs_data['run1'] else 0
            last_count_run2 = runs_data['run2'][-1] if runs_data['run2'] else 0
            last_count_run3 = runs_data['run3'][-1] if runs_data['run3'] else 0
            combined_total = last_count_run1 + last_count_run2 + last_count_run3
            
            line = f"| {display_name:<11} | {last_count_run1:<16} | {last_count_run2:<16} | {last_count_run3:<16} | {combined_total:<14} |"
            f.write(f"{line}\n")
    
    print(f"Last round results with separate runs saved to {last_round_filename}")
    
    # Create figure for muformer only
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Define method name mapping for display
    method_display_names = {
        'musearch_gt': 'μSearch',
        'adalead_gt': 'AdaLead',
        'cmaes_gt': 'CMA-ES',
        'BO_gt': 'BO',
        'cbas_gt': 'CbAS',
        'dynappo_gt': 'DyNA-PPO',
        'evoplay_gt': 'EvoPlay',
        'gwg_gt': 'GWG',
        'pure_random': 'Random',
        'pex_gt': 'PEX'
    }

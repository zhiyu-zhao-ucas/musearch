import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
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

def count_peaks_above_threshold(landscape, methods, threshold, max_fitness, min_fitness, sequences_batch_size=100, model_queries_per_batch=5000, max_round=10):
    """
    Count how many data points exceed the threshold after normalization.
    Normalization: (score - min_fitness) / (max_fitness - min_fitness)
    Threshold is applied to the normalized value.
    """
    results = {}
    # Avoid division by zero
    fitness_range = max_fitness - min_fitness
    if fitness_range == 0:
        fitness_range = 1
        
    # Threshold is now applied to normalized values
    normalized_threshold = threshold  # Threshold is directly applied to normalized scores (0 to 1)
    
    for method in methods:
        method_data = defaultdict(list)  # To store counts for each round
        
        directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
        pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
        
        csv_files = get_all_csv_files(directory, pattern)
        
        for file_path in csv_files:
            try:
                with open(file_path) as f:
                    metadata = json.loads(next(f))  # Skip metadata line
                    data = pd.read_csv(f)
                    
                    # Count data points over threshold in each round
                    num_rounds = min(data['round'].max() + 1, max_round + 1)
                    
                    for r in range(num_rounds):
                        round_data = data[data['round'] <= r]  # Consider only data up to round r
                        
                        # Normalize fitness scores and count points exceeding threshold
                        normalized_scores = (round_data['true_score'] - min_fitness) / fitness_range
                        points_over_threshold = sum(normalized_scores >= normalized_threshold) if not round_data.empty else 0
                        method_data[r].append(points_over_threshold)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Calculate the total count for each round across all runs
        peaks_per_round = []
        for r in sorted(method_data.keys()):
            if r <= max_round:  # Only include rounds up to max_round
                count = sum(method_data[r])
                peaks_per_round.append(count)
        
        results[method] = peaks_per_round
    
    return results

def analyze_landscape(landscape, threshold=0.95, sequences_batch_size=100, model_queries_per_batch=5000, max_round=10):
    """Analyze a single landscape for peak counts"""
    print(f"Analyzing landscape: {landscape}")
    
    # Method list for muformer (ground truth landscape)
    method_list = ['pure_random', 'adalead_gt', 'BO_gt', 'musearch_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'gwg_gt']
    
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
        'pure_random': 'Random'
    }
    
    # Find max and min fitness for this landscape
    max_fitness, min_fitness = find_max_min_fitness_for_landscape(landscape, method_list, sequences_batch_size, model_queries_per_batch, max_round)
    print(f"Fitness range for {landscape} up to round {max_round}: min={min_fitness}, max={max_fitness}")
    
    # Count peaks for each method
    peak_counts = count_peaks_above_threshold(landscape, method_list, threshold, max_fitness, min_fitness,
                                              sequences_batch_size, model_queries_per_batch, max_round)
    
    # Print and plot results
    plt.figure(figsize=(10, 6), dpi=300)
    
    for method, counts in peak_counts.items():
        display_name = method_display_names.get(method, method)
        
        # Get total number of data points per round (for percentage calculation)
        directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
        pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
        csv_files = get_all_csv_files(directory, pattern)
        
        # Calculate total number of sequences evaluated per round
        total_sequences_per_round = 0
        for file_path in csv_files:
            try:
                with open(file_path) as f:
                    metadata = json.loads(next(f))  # Skip metadata line
                    data = pd.read_csv(f)
                    if not data.empty:
                        # For the first round, count sequences to know how many per round
                        first_round = data[data['round'] == 0]
                        total_sequences_per_round += len(first_round)
            except Exception as e:
                print(f"Error calculating total sequences: {e}")
        
        if total_sequences_per_round > 0 and len(counts) > 0:
            print(f"{display_name}: {counts} data points exceed {threshold * 100}% of normalized max fitness")
            
            # Plot results
            rounds = np.arange(len(counts)) * model_queries_per_batch
            plt.plot(rounds, counts, '-o', label=display_name, markersize=3)
    
    plt.title(f'Number of data points exceeding {threshold * 100}% of normalized max fitness on {landscape}', fontweight='bold')
    plt.xlabel('Number of Model Queries', fontweight='bold')
    plt.ylabel('Number of Data Points', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    
    # Save figure
    os.makedirs('examples/figs/peak_counts', exist_ok=True)
    plt.savefig(f'examples/figs/peak_counts/normalized_points_over_threshold_{landscape}_{int(threshold * 100)}_round_{max_round}.png')
    print(f"Analysis complete. Results saved to examples/figs/peak_counts/normalized_points_over_threshold_{landscape}_{int(threshold * 100)}_round_{max_round}.png")
    true_threshold = (max_fitness - min_fitness) * threshold + min_fitness
    print(f"Threshold: {true_threshold}")
    
    return max_fitness, min_fitness, peak_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.90, 
                        help='Threshold fraction of normalized max fitness to count as a peak (default: 0.95)')
    # parser.add_argument('--threshold', type=float, default=0.8814250112466141, 
    #                     help='Threshold fraction of normalized max fitness to count as a peak (default: 0.95)')
    parser.add_argument('--max_round', type=int, default=50,
                        help='Maximum round to consider (default: 10)')
    args = parser.parse_args()
    
    # Only analyze muformer landscape
    landscape = 'muformer'
    
    # Analyze the landscape
    max_fitness, min_fitness, peak_counts = analyze_landscape(landscape, args.threshold, max_round=args.max_round)
    
    # Save results to text file
    results_filename = f'examples/figs/peak_counts/results_{landscape}_{int(args.threshold * 100)}_round_{args.max_round}.md'
    with open(results_filename, 'w') as f:
        f.write(f"# Analysis for landscape: {landscape}\n\n")
        f.write(f"- **Threshold**: {args.threshold * 100}% of normalized max fitness\n")
        f.write(f"- **Max round**: {args.max_round}\n")
        f.write(f"- **Fitness range**: min={min_fitness}, max={max_fitness}\n")
        true_threshold = (max_fitness - min_fitness) * args.threshold + min_fitness
        f.write(f"- **Absolute threshold value**: {true_threshold}\n\n")
        
        f.write("## Data points exceeding threshold by method and round\n\n")
        
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
            'pure_random': 'Random'
        }
        
        # Create a table header with rounds
        rounds = [r for r in range(len(next(iter(peak_counts.values()))) if peak_counts else 0)]
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
        
        # Write data for each method
        for method, counts in peak_counts.items():
            display_name = method_display_names.get(method, method)
            line = f"| {display_name:<11}"
            for count in counts:
                line += f"| {count:<16}"
            line += "|"
            f.write(f"{line}\n")
    
    print(f"Results saved to {results_filename}")
    
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
        'pure_random': 'Random'
    }
    
    # Method list for muformer
    method_list = ['pure_random', 'adalead_gt', 'BO_gt', 'musearch_gt', 'cbas_gt', 'evoplay_gt', 'dynappo_gt', 'cmaes_gt', 'gwg_gt']
    
    # Print the summary table of results to console
    print("\nSummary of data points exceeding threshold by method and round:")
    print("------------------------------------------------")
    
    # Create a table header with rounds
    rounds = [r * 5000 for r in range(len(next(iter(peak_counts.values()))) if peak_counts else 0)]
    header = "Method        "
    for r in rounds:
        header += f" | Round {r:<6}"
    print(f"{header}")
    print("-" * len(header))
    
    # Print data for each method
    for method in method_list:
        if method in peak_counts:
            counts = peak_counts[method]
            display_name = method_display_names.get(method, method)
            line = f"{display_name:<13}"
            for count in counts:
                line += f" | {count:<12}"
            print(f"{line}")
    
    # Plot for each method
    for method in method_list:
        if method in peak_counts:
            counts = peak_counts[method]
            
            # Get total sequences per round
            directory = f'/home/v-zhaozhiyu/code/musearch/efficiency/{method}/{landscape}'
            pattern = "100_5000"
            csv_files = get_all_csv_files(directory, pattern)
            
            # Calculate total number of sequences evaluated per round
            total_sequences_per_round = 0
            for file_path in csv_files:
                try:
                    with open(file_path) as f:
                        metadata = json.loads(next(f))  # Skip metadata line
                        data = pd.read_csv(f)
                        if not data.empty:
                            # For the first round, count sequences to know how many per round
                            first_round = data[data['round'] == 0]
                            total_sequences_per_round += len(first_round)
                except Exception as e:
                    print(f"Error calculating total sequences: {e}")
            
            if total_sequences_per_round > 0 and len(counts) > 0:
                display_name = method_display_names.get(method, method)
                rounds = np.arange(len(counts)) * 5000
                plt.plot(rounds, counts, '-o', label=display_name, markersize=3)
    
    plt.title(f'Number of data points exceeding {args.threshold * 100}% of normalized max fitness on {landscape}', fontweight='bold')
    plt.xlabel('Number of Model Queries', fontweight='bold')
    plt.ylabel('Number of Data Points', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'examples/figs/peak_counts/muformer_normalized_{int(args.threshold * 100)}_round_{args.max_round}.png')
    
    print(f"Analysis complete. Results saved to examples/figs/peak_counts/")

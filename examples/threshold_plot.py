import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
import re
from datetime import datetime


WT_fitness = -0.0726749
ESBL_fitness_median = -0.0646437555551529

def cumulative_max_per_round(sequences):
    print(f"Calculating cumulative max per round for {len(sequences)} sequences")
    num_rounds = sequences['round'].max() + 1
    print(f"Number of rounds: {num_rounds}")
    for r in range(num_rounds):
        print(f"Round {r}: {sequences['true_score'][sequences['round'] == r].max()}")
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]
    print(f"Max per round: {max_per_round}")

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
    
    return mean_values, std_values, min_length, all_runs_array


def find_first_round_exceeding_threshold(values, threshold):
    """Find the first round where the value exceeds or equals the threshold"""
    indices = np.where(values >= threshold)[0]
    if len(indices) > 0:
        return indices[0]
    return np.nan  # Return NaN if threshold is never reached


def create_threshold_comparison_df(musearch_values, random_values, query_multiplier=5000):
    """
    Create a DataFrame comparing when each method first reaches threshold values.
    
    Args:
        musearch_values: 2D array of values for musearch_gt (runs x rounds)
        random_values: 2D array of values for pure_random (runs x rounds)
        query_multiplier: Number of queries per round
    
    Returns:
        DataFrame with threshold values and rounds to reach them
    """
    # Calculate average performance per round for each method
    musearch_avg = np.mean(musearch_values, axis=0)
    random_avg = np.mean(random_values, axis=0)
    
    # Get unique threshold values from musearch_gt's performance
    # We use unique values to avoid redundant rows
    thresholds = np.linspace(1.0, 19, num=5)
    # add random_avg.max into thresholds
    thresholds = np.unique(np.concatenate((thresholds, [random_avg.max()])))
    thresholds = np.sort(thresholds)

    
    # Initialize results list
    results = []
    
    # For each threshold value
    for threshold in thresholds:
        # Find first round where each method exceeds the threshold
        musearch_round = np.float16(find_first_round_exceeding_threshold(musearch_avg, threshold))
        print(f"musearch_avg: {musearch_avg}")
        random_round = np.float16(find_first_round_exceeding_threshold(random_avg, threshold))
        print(f"random_avg: {random_avg}")
        
        # Convert rounds to queries if multiplier is provided
        musearch_queries = musearch_round * query_multiplier if not np.isnan(musearch_round) else np.nan
        random_queries = random_round * query_multiplier if not np.isnan(random_round) else np.nan
        
        results.append({
            "normalized_reward": threshold,
            "raw_reward": threshold * (ESBL_fitness_median - WT_fitness) + WT_fitness,
            # "musearch_gt min round": musearch_round,
            # "pure_random min round": random_round,
            "musearch_gt min queries": musearch_queries,
            "pure_random min queries": random_queries,
            "M/N": random_queries / musearch_queries
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', action='store_true')
    args = parser.parse_args()
    
    if args.cnn:
        landscapes = ['aav', 'rna', 'gfp', 'tf', 'rosetta']
    else:
        landscapes = ['muformer']
        
    for landscape in landscapes:
        for sequences_batch_size, model_queries_per_batch in [(5, 50)]:
            print(f"\nAnalyzing {landscape} with {sequences_batch_size} sequences, {model_queries_per_batch} queries per batch\n")
            
            # Get data for musearch_gt
            musearch_dir = f'/home/v-zhaozhiyu/code/musearch/efficiency/musearch_gt/{landscape}'
            musearch_pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
            _, _, _, musearch_values = calculate_stats_across_runs(musearch_dir, musearch_pattern)
            
            # Get data for pure_random
            random_dir = f'/home/v-zhaozhiyu/code/musearch/efficiency/pure_random/{landscape}'
            random_pattern = f"{sequences_batch_size}_{model_queries_per_batch}"
            _, _, _, random_values = calculate_stats_across_runs(random_dir, random_pattern)
            
            # Create and print comparison DataFrame
            comparison_df = create_threshold_comparison_df(
                musearch_values, 
                random_values, 
                query_multiplier=model_queries_per_batch
            )
            
            print("\nThreshold Comparison:")
            print(comparison_df.to_string())
            
            # Save DataFrame to CSV
            output_dir = f'examples/threshold_data'
            os.makedirs(output_dir, exist_ok=True)
            output_file = f'{output_dir}/threshold_comparison_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.csv'
            comparison_df.to_csv(output_file, index=False)
            print(f"\nDataFrame saved to: {output_file}")
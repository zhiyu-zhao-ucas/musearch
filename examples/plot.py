import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np


def cumulative_max_per_round(sequences):
    num_rounds = sequences['round'].max() + 1
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]

    return np.maximum.accumulate(max_per_round)

for landscape in ['aav', 'rna', 'gfp', 'tf', 'rosetta']:
    for sequences_batch_size, model_queries_per_batch in [(100, 5000)]:
        plt.figure(dpi=300)
        plt.title(f'Performance with respect to {sequences_batch_size} and {model_queries_per_batch} on {landscape}')
        for method in ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO', 'dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
            if method in ['dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}/{sequences_batch_size}_{model_queries_per_batch}_new.csv'
            else:
                file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}/{sequences_batch_size}_{model_queries_per_batch}.csv'
            with open(file_name) as f:
                metadata = json.loads(next(f))
                data = pd.read_csv(f)

                rounds = data['round'].unique()
                max_per_round = cumulative_max_per_round(data)
            try:
                plt.plot(rounds, max_per_round, '-o', label=f'{method}')
            except:
                pass
            plt.ylabel("Cumulative max")
            plt.xlabel("Round")

        plt.legend()
        plt.savefig(f'examples/figs/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.png')
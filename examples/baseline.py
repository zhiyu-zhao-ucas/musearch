import editdistance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pprint
import numpy as np
import json
import flexs
from flexs import baselines
import flexs.landscapes.additive_aav_packaging
import flexs.utils.sequence_utils as s_utils


problem = flexs.landscapes.rna.registry()['L14_RNA1']
# problem = flexs.landscapes.additive_aav_packaging.registry()['blood']
print(problem)
pprint.pprint(problem)
starting_sequence = problem['starts'][0]
print(problem['params'])
landscape = flexs.landscapes.RNABinding(**problem['params'])
alphabet = s_utils.RNAA
def make_explorer(sequences_batch_size, model_queries_per_batch):
    model = baselines.models.NoisyAbstractModel(landscape, signal_strength=1)
    return baselines.explorers.Adalead(
            model,
            rounds=5,
            mu=1,
            starting_sequence=starting_sequence,
            sequences_batch_size=sequences_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet,
            log_file=f'efficiency/adalead/{sequences_batch_size}_{model_queries_per_batch}.csv',
        )

results = flexs.evaluate.efficiency(landscape, make_explorer, budgets=[(100, 500), (100, 5000),(1000, 5000),(1000, 10000)])
print(results)
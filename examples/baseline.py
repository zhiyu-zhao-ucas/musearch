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
import argparse

def main(args):
    if args.landscape == 'aav':
        problem = flexs.landscapes.additive_aav_packaging.registry()['blood']
        pprint.pprint(problem)
        print(problem['params'])
        landscape = flexs.landscapes.additive_aav_packaging.AdditiveAAVPackaging(**problem['params'])
        starting_sequence = landscape.wild_type
        alphabet = s_utils.AAS
    elif args.landscape == 'rna':
        problem = flexs.landscapes.rna.registry()['L14_RNA1']
        pprint.pprint(problem)
        print(problem['params'])
        starting_sequence = problem['starts'][0]
        landscape = flexs.landscapes.RNABinding(**problem['params'])
        alphabet = s_utils.RNAA
    elif args.landscape == 'gfp':
        landscape = flexs.landscapes.BertGFPBrightness()
        starting_sequence = landscape.gfp_wt_sequence
        alphabet = s_utils.AAS
    elif args.landscape == 'tf':
        problem = flexs.landscapes.tf_binding.registry()['SIX6_REF_R1']
        pprint.pprint(problem)
        landscape = flexs.landscapes.TFBinding(**problem['params'])
        starting_sequence = problem['starts'][0]
        alphabet = s_utils.DNAA
    elif args.landscape == 'rosetta':
        problem = flexs.landscapes.rosetta.registry()['3mx7']
        landscape = flexs.landscapes.RosettaFolding(**problem['params'])
        starting_sequence = landscape.wt_pose.sequence()
        alphabet = s_utils.AAS
    else:
        raise ValueError('Unknown landscape')
    def make_explorer(sequences_batch_size, model_queries_per_batch):
        if args.method == 'adalead':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.Adalead(
                    cnn,
                    rounds=5,
                    mu=1,
                    starting_sequence=starting_sequence,
                    sequences_batch_size=sequences_batch_size,
                    model_queries_per_batch=model_queries_per_batch,
                    alphabet=alphabet,
                    log_file=f'efficiency/{args.method}/{args.landscape}/{sequences_batch_size}_{model_queries_per_batch}.csv',
                )
        elif args.method == 'cmaes':
            return baselines.explorers.CMAES(
                flexs.LandscapeAsModel(landscape),
                
                population_size=10,
                max_iter=200,
                
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=100,
                model_queries_per_batch=1000,
                alphabet=alphabet
            )
        elif args.method == 'dynappo':
            return baselines.explorers.DynaPPO(  # DynaPPO has its own default ensemble model, so don't use CNN
                    landscape=landscape,
                    env_batch_size=10,
                    num_model_rounds=10,
                    rounds=10,
                    starting_sequence=starting_sequence,
                    sequences_batch_size=100,
                    model_queries_per_batch=1000,
                    alphabet=alphabet,
                )
        elif args.method == 'cbas':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')

            vae = baselines.explorers.VAE(len(starting_sequence), alphabet=alphabet, epochs=10, verbose=False)
            return baselines.explorers.CBAS(
                flexs.LandscapeAsModel(landscape),
                vae,
                cnn,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=100,
                model_queries_per_batch=1000,
                alphabet=alphabet
            )
        elif args.method == 'BO':
            cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                         num_filters=32, hidden_size=100, loss='MSE')
            return baselines.explorers.BO(
                model=cnn,
                rounds=10,
                starting_sequence=starting_sequence,
                sequences_batch_size=100,
                model_queries_per_batch=1000,
                alphabet=alphabet,
            )
        elif args.method == 'gwg':
            encoder = flexs.baselines.explorers.Encoder(alphabet)
            sampler = flexs.baselines.explorers.GwgPairSampler(encoder, 10, sequences_batch_size=10, model_queries_per_batch=10, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet, log_file=f'efficiency/{args.method}/{args.landscape}/10_10.csv')
            return flexs.baselines.explorers.GWG(model=sampler, rounds=10, sequences_batch_size=10, model_queries_per_batch=10, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet)



    results = flexs.evaluate.efficiency(landscape, make_explorer, budgets=[(100, 500), (100, 5000),(1000, 5000),(1000, 10000)])
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='adalead')
    parser.add_argument('--landscape', type=str, default='rna')
    args = parser.parse_args()
    main(args)

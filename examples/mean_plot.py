import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob


def cumulative_max_per_round(sequences):
    num_rounds = sequences['round'].max() + 1
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]

    return np.maximum.accumulate(max_per_round)


def get_latest_csv(directory, name_pattern, num_files=3, exclude=None):
    """获取指定目录下最新的CSV文件
    优先使用带时间戳的文件，如果没有则使用最后修改时间"""
    # 获取目录下所有csv文件
    pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"在 {directory} 目录下没有找到CSV文件")
    
    # 尝试找出带有时间戳格式的文件
    timestamped_files = []
    other_files = []
    
    for file in csv_files:
        try:
            # 尝试解析文件名中的时间戳
            # print(f"file: {file}, name_pattern: {name_pattern}")
            if name_pattern in file:
                if exclude is not None:
                    exclude_list = [ex for ex in exclude if ex in file]
                    if any(exclude_list):
                        continue
                timestamped_files.append(file)
        except (IndexError, ValueError):
            # 如果解析失败，将文件添加到其他文件列表
            # print(f"file: {file}, name_pattern: {name_pattern}")
            if name_pattern in file:
                if exclude is not None:
                    exclude_list = [ex for ex in exclude if ex in file]
                    if any(exclude_list):
                        continue
                other_files.append(file)
    
    if timestamped_files:
        # 如果有带时间戳的文件，选择时间戳最新的
        sorted_files = sorted(timestamped_files, 
                    key=lambda x: os.path.basename(x).split('_')[-1].split('.')[0],
                    reverse=True)
        latest_file = sorted_files[:num_files]
    else:
        # 如果没有带时间戳的文件，使用最后修改时间最新的文件
        try:
            latest_file = max(other_files, key=os.path.getmtime)
            sorted_files = sorted(other_files,
                    key=os.path.getmtime,
                    reverse=True)
            latest_file = sorted_files[:num_files]
        except:
            print(directory)
        print("使用最后修改时间最新的文件")
    
    return latest_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', action='store_true')
    args = parser.parse_args()
    # for landscape in ['muformer']:
    if args.cnn:
        landscapes = ['aav', 'rna', 'gfp', 'tf', 'rosetta']
    else:
        landscapes = ['muformer']
    for landscape in landscapes:
        for sequences_batch_size, model_queries_per_batch in [(100, 5000)]:
            plt.figure(dpi=300)
            plt.title(f'Performance with respect to {sequences_batch_size} and {model_queries_per_batch} on {landscape}')
            if args.cnn:
                method_list = ['adalead', 'cmaes', 'BO', 'cbas', 'dynappo', 'dirichlet_ppo_cnn', 'pex', 'gwg', 'evoplay']
                # method_list = ['adalead', 'cmaes', 'BO', 'cbas', 'dynappo', 'dirichlet_ppo_cnn']
                # method_list = ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO']
                dirichlet_ppo_list = ['dirichlet_ppo_cnn']
                # dirichlet_ppo_list = ['dirichlet_ppo_cnn']
            else:
                method_list = ['adalead_gt', 'cmaes_gt', 'cbas_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo', 'pex_gt', 'evoplay_gt']
                # dirichlet_ppo_list = []
                dirichlet_ppo_list = ['dirichlet_ppo']
            for method in method_list:
                print(f"method: {method}")
            # for method in ['adalead_gt', 'cbas_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                if method in dirichlet_ppo_list:
                # if method in ['dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                    if args.cnn:
                        if landscape != 'rna':
                            file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch)+"_starting_sequence", num_files=3)
                        else:
                            file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch)+"_freq_3", num_files=3)
                    else:
                        file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch)+'_freq_5', num_files=3)
                        # file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch)+"_alpha05", num_files=3)
                        # file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch)+'_alpha05', num_files=3, exclude=['freq_5'])
                else:
                    file_names = get_latest_csv(f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch), num_files=3)
                # print(f"file_names: {file_names}")
                max_per_rounds = []
                for file_name in file_names:
                    with open(file_name) as f:
                        metadata = json.loads(next(f))
                        data = pd.read_csv(f)

                        rounds = data['round'].unique()
                        max_per_round = cumulative_max_per_round(data)
                        # check whether max_per_round including nan
                        if np.isnan(max_per_round).any():
                            print(f"nan in {file_name}")
                        max_per_rounds.append(max_per_round)
                try:
                    print(f"lens: {min([len(max_per_round) for max_per_round in max_per_rounds])}")
                    min_len = min([len(max_per_round) for max_per_round in max_per_rounds])
                    rounds = np.arange(min_len)
                    if not args.cnn:
                        rounds = rounds * 5000
                    print(f"rounds: {rounds}, min_len: {min_len}")
                    max_per_rounds = np.array([max_per_round[:min_len] for max_per_round in max_per_rounds])
                    max_per_round = np.mean(max_per_rounds, axis=0)
                    max_per_round_std = np.std(max_per_rounds, axis=0)
                    print(f"max_per_round[-1]: {max_per_round[-1]}, max_per_round_std[-1]: {max_per_round_std[-1]}")
                    plt.plot(rounds, max_per_round, '-o', label=f'{method}')
                except Exception as e:
                    print(file_name)
                    # print(f"rounds: {rounds}, max_per_round: {max_per_round}")
                    print(e)                
                    pass
                plt.ylabel("Cumulative max")
                plt.xlabel("Number of measured sequences")

            plt.legend()
            # plt.savefig(f'examples/figs/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.png')
            if args.cnn:
                plt.savefig(f'examples/figs/cnn/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_cnn_freq3.png')
            else:
                plt.savefig(f'examples/figs/gt/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_gt_freq5.png')
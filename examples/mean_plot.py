import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
import re


def cumulative_max_per_round(sequences):
    num_rounds = sequences['round'].max() + 1
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]

    return np.maximum.accumulate(max_per_round)


def get_latest_csv(directory, name_pattern, num_files=3, exclude=None):
    """
    获取目录中符合指定模式的CSV文件。
    
    仅匹配文件名格式为 {name_pattern}_1.csv, {name_pattern}_2.csv, {name_pattern}_3.csv，
    排除其它格式如 {name_pattern}.csv, {name_pattern}_0.csv 或 {name_pattern}_4.csv 等。
    
    参数:
        directory (str): 查找CSV文件的目录。
        name_pattern (str): 文件名前缀，如 "xxx"。
        num_files (int): 返回的文件数量（默认3）。
        exclude (list or None): 如果提供，在文件名中包含列表中任意字符串的文件将被排除。
    
    返回:
        list: 符合条件的文件路径列表（按文件名中数字升序排序）。
    """
    # 构建正则表达式，匹配 {name_pattern}_数字.csv，其中数字必须为 1、2 或 3
    regex = re.compile(rf"^{re.escape(name_pattern)}_([123])\.csv$")
    
    # 获取目录下所有CSV文件
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {directory} 目录下没有找到CSV文件")
    
    matched_files = []
    for file in csv_files:
        base = os.path.basename(file)
        # 仅匹配符合格式的文件
        if regex.match(base):
            if exclude is not None and any(ex in base for ex in exclude):
                continue
            matched_files.append(file)
    
    if not matched_files:
        raise FileNotFoundError(f"在 {directory} 目录下没有找到符合条件的CSV文件")
    
    # 根据文件名中下划线后的数字进行排序（例如：xxx_1, xxx_2, xxx_3）
    sorted_files = sorted(matched_files, 
                          key=lambda x: int(re.search(r'_([123])\.csv$', os.path.basename(x)).group(1)))
    
    return sorted_files[:num_files]


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
                # method_list = ['adalead', 'cmaes', 'BO', 'dynappo', 'dirichlet_ppo_cnn', 'pex', 'gwg', 'evoplay']
                method_list = ['adalead', 'cmaes', 'BO', 'cbas', 'dynappo', 'dirichlet_ppo_cnn', 'pex', 'gwg', 'evoplay']
                # method_list = ['adalead', 'cmaes', 'BO', 'cbas', 'dynappo', 'dirichlet_ppo_cnn']
                # method_list = ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO']
                dirichlet_ppo_list = ['dirichlet_ppo_cnn']
                # dirichlet_ppo_list = ['dirichlet_ppo_cnn']
            else:
                # method_list = ['adalead_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo', 'pex_gt', 'evoplay_gt']
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
                    print(f"method: {method}, landscape: {landscape}")
                    print(f"max_per_round[-1], (max_per_round_std[-1]): {max_per_round[-1]} ({max_per_round_std[-1]})")
                    if "dirichlet_ppo" in method:
                        plt.plot(rounds, max_per_round, '-o', label=f'our')
                    elif "_gt" in method:
                        plt.plot(rounds, max_per_round, '-o', label=f'{method.replace("_gt", "")}')
                    else:
                        plt.plot(rounds, max_per_round, '-o', label=f'{method}')
                except Exception as e:
                    print(file_name)
                    # print(f"rounds: {rounds}, max_per_round: {max_per_round}")
                    print(e)                
                    pass
                plt.ylabel("Cumulative max")
                if not args.cnn:
                    plt.xlabel("Number of measured sequences")
                else:
                    plt.xlabel("Rounds")

            plt.legend()
            # plt.savefig(f'examples/figs/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.png')
            if args.cnn:
                plt.savefig(f'examples/figs/cnn/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.pdf')
            else:
                plt.savefig(f'examples/figs/gt/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.pdf')
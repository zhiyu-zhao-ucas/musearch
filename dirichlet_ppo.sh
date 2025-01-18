#!/bin/bash

# 创建新的 tmux session
tmux new-session -d -s our

landscapes=(aav gfp rna tf rosetta)
methods=(dirichlet_ppo_cnn dirichlet_ppo)

gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
window_count=0
for landscape in "${landscapes[@]}"; do
    for method in "${methods[@]}"; do
        # 为每个命令创建新窗口
        gpu_index=$((window_count % num_gpus))
        current_gpu=${gpus[$gpu_index]}
        if [ $window_count -eq 0 ]; then
            # 第一个窗口已经存在，重命名它
            tmux rename-window "${method}_${landscape}"
        else
            # 创建新窗口
            tmux new-window -n "${method}_${landscape}"
        fi
        
        # 在窗口中运行命令
        tmux send-keys "conda activate nmi" C-m
        tmux send-keys "export CUDA_VISIBLE_DEVICES=$current_gpu" C-m
        tmux send-keys "echo 'Using GPU $current_gpu'" C-m
        tmux send-keys "python examples/baseline.py --method $method --landscape $landscape" C-m
        
        ((window_count++))
    done
done

# 切换到第一个窗口
tmux select-window -t 0

# 附加到 session
tmux attach-session -t our

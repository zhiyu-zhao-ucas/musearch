#!/bin/bash

# 创建新的 tmux session
tmux new-session -d -s evoplay

landscapes=(gfp tf aav rosetta rna)
# landscapes=(muformer)
methods=(evoplay)
# methods=(pex_gt)
# methods=(adalead cmaes BO dynappo cbas)
# methods=(adalead cmaes dynappo cbas BO dirichlet_ppo_cnn)

gpus=(0 1 2 3)
num_gpus=${#gpus[@]}
window_count=0

for landscape in "${landscapes[@]}"; do
    for method in "${methods[@]}"; do
        for run in {1..3}; do
            # 为每个命令创建新窗口
            gpu_index=$((window_count % num_gpus))
            current_gpu=${gpus[$gpu_index]}
            
            window_name="${method}_${landscape}_run${run}"
            
            if [ $window_count -eq 0 ]; then
                # 第一个窗口已经存在，重命名它
                tmux rename-window "$window_name"
            else
                # 创建新窗口
                tmux new-window -n "$window_name"
            fi
            
            # 在窗口中运行命令
            tmux send-keys "conda activate FLEXS" C-m
            tmux send-keys "export CUDA_VISIBLE_DEVICES=$current_gpu" C-m
            tmux send-keys "echo 'Using GPU $current_gpu for $window_name'" C-m
            # 添加随机种子以确保每次运行都不同
            tmux send-keys "python examples/baseline.py --method $method --landscape $landscape --run $run" C-m
            
            ((window_count++))
        done
    done
done

# 切换到第一个窗口
tmux select-window -t 0

# 附加到 session
tmux attach-session -t evoplay
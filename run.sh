#!/bin/bash

# 创建新的 tmux session
tmux new-session -d -s baseline

landscapes=(aav gfp rna tf rosetta)
methods=(adalead_gt cmaes_gt dynappo_gt cbas_gt BO_gt)

window_count=0
for landscape in "${landscapes[@]}"; do
    for method in "${methods[@]}"; do
        # 为每个命令创建新窗口
        if [ $window_count -eq 0 ]; then
            # 第一个窗口已经存在，重命名它
            tmux rename-window "${method}_${landscape}"
        else
            # 创建新窗口
            tmux new-window -n "${method}_${landscape}"
        fi
        
        # 在窗口中运行命令
        tmux send-keys "conda activate nmi" C-m
        tmux send-keys "python examples/baseline.py --method $method --landscape $landscape" C-m
        
        ((window_count++))
    done
done

# 切换到第一个窗口
tmux select-window -t 0

# 附加到 session
tmux attach-session -t experiments
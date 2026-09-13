#!/bin/bash
PROJECT_ROOT="/home/felix/isaac/xMimic/xGMR"
MOTION_DIR="/home/felix/isaac/xMimic/motion_npz/bvh_npz"
TASK="Tracking-Flat-DexEVT-Wo-State-Estimation-v0"
# 替换为你阶段1训练好的权重路径
CKPT="/home/felix/isaac/xMimic/logs/rsl_rl/Tracking-Flat-DexEVT-Wo-State-Estimation-v0/2026-07-04_10-00-00/model.pt"
cd ${PROJECT_ROOT}
NPZ_LIST=(${MOTION_DIR}/*.npz)
for npz in "${NPZ_LIST[@]}"; do
    name=$(basename $npz .npz)
    echo "==== Stage2 混合微调：$name ===="
    PYTHONPATH=. python scripts/rsl_rl/train.py \
        --task ${TASK} \
        --num_envs 1024 \
        --max_iterations 2000 \
        --device cuda:0 \
        --load_checkpoint ${CKPT} \
        --motion_file "$npz" \
        --headless \
        --logger tensorboard \
        --log_dir /home/felix/isaac/xMimic/train_logs
    echo -e "===== ${name} 训练完成 =====\n"
done


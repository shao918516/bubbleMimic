#!/bin/bash
set -euo pipefail
PROJECT_ROOT="/home/felix/isaac/xMimic"
MOTION_DIR="/home/felix/isaac/xMimic/motion_data/bvh_npz"
TASK_NAME="Tracking-Flat-DexEVT-Wo-State-Estimation-v0"
NUM_ENVS=1024
MAX_ITER=200
DEVICE="cuda:0"
LOG_ROOT="/home/felix/isaac/xMimic/train_logs/stage1_g1_all"

cd "${PROJECT_ROOT}"
# 获取所有npz
NPZ_LIST=("${MOTION_DIR}"/*.npz)
TOTAL=${#NPZ_LIST[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "目录无npz动作文件"
    exit 1
fi
echo "共${TOTAL}个G1动作，开始递进训练"

# 初始权重为空，第一条随机初始化
CUR_CKPT="/home/felix/isaac/xMimic/logs/rsl_rl/dex_evt_flat/2026-07-05_13-42-08/model_999.pt"

for npz_path in "${NPZ_LIST[@]}"; do
    motion_name=$(basename "${npz_path}" .npz)
    echo -e "\n==== 当前动作：${motion_name} | 加载权重：${CUR_CKPT:-随机初始化} ===="

    CMD=(
        python scripts/rsl_rl/train.py
        --task "${TASK_NAME}"
        --num_envs "${NUM_ENVS}"
        --max_iterations "${MAX_ITER}"
        --device "${DEVICE}"
        --motion_file "${npz_path}"
        --headless
        --logger tensorboard
        --run_name "${motion_name}"
    )
    if [[ -n "${CUR_CKPT}" ]]; then
        CMD+=(--checkpoint "${CUR_CKPT}")
    fi

    PYTHONPATH=. "${CMD[@]}"

    # 全局搜索本次动作对应的所有model_*.pt
    NEW_CKPT=$(find /home/felix/isaac/xMimic/logs/rsl_rl/dex_evt_flat -path "*_${motion_name}/model_*.pt" -type f | grep -E "model_[0-9]+\.pt" | sort -t'_' -k2 -n | tail -n 1)

    if [[ -z "${NEW_CKPT}" ]]; then
        echo "警告：${motion_name} 未找到权重文件，终止脚本"
        exit 1
    fi
    CUR_CKPT="${NEW_CKPT}"
    echo "===== ${motion_name} 训练完成，下一轮加载权重：${CUR_CKPT} ====\n"
done

echo "全部动作训练完成，最终权重：${CUR_CKPT}"


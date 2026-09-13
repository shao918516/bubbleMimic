#!/bin/bash
set -euo pipefail

# 全局配置
PROJECT_ROOT="/home/felix/isaac/xMimic"
MOTION_DATA_DIR="/home/felix/isaac/xMimic/motion_npz/g1_npz"
TASK_NAME="Tracking-Flat-DexEVT-Wo-State-Estimation-v0"
NUM_ENVS=1024
MAX_ITER=2000
DEVICE="cuda:0"
LOGGER="tensorboard"

# 进入项目根目录
cd "${PROJECT_ROOT}" || { echo "无法进入项目根目录"; exit 1; }

# 获取所有npz文件
NPZ_LIST=("${MOTION_DATA_DIR}"/*.npz)
TOTAL=${#NPZ_LIST[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "motion_data 下无npz动作文件，退出"
    exit 1
fi

echo "====================================="
echo "共检测到 ${TOTAL} 个动作文件，开始批量训练"
echo "====================================="

for npz_path in "${NPZ_LIST[@]}"; do
    motion_name=$(basename "${npz_path}" .npz)
    echo -e "\n===== 基础训练动作：${motion_name} ====="
    PYTHONPATH=. python scripts/rsl_rl/train.py \
        --task ${TASK_NAME} \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITER} \
        --device ${DEVICE} \
        --motion_file "${npz_path}" \
        --headless \
        --logger ${LOGGER}
        --log_dir /home/felix/isaac/xMimic/train_logs
    echo -e "===== ${motion_name} 训练完成 =====\n"
done

echo "全部动作训练完毕！"


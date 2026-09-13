#!/bin/bash
set -euo pipefail

# ====================== 配置区======================
PROJECT_ROOT="/home/felix/isaac/xMimic"
PKL_FOLDER="${PROJECT_ROOT}/xGMR/motion_pkl"
ROBOT_TYPE="dex_evt"
INPUT_FPS=30
OUTPUT_FPS=100
FRAME_START=10
FRAME_END=-1
START_FRAMES=100
END_FRAMES=50
HOLD_POS=300
OUTPUT_DIR="${PROJECT_ROOT}/xGMR/motion_npz"
# ==============================================================

# 进入项目根目录
cd "${PROJECT_ROOT}" || { echo "错误：无法进入项目目录 ${PROJECT_ROOT}"; exit 1; }

# 创建输出文件夹
mkdir -p "${OUTPUT_DIR}"

# 获取所有pkl文件
PKL_LIST=("${PKL_FOLDER}"/*.pkl)
TOTAL_COUNT=${#PKL_LIST[@]}

# 判断是否存在pkl
if [[ ${TOTAL_COUNT} -eq 0 ]]; then
    echo "错误：${PKL_FOLDER} 下未找到任何 .pkl 文件"
    exit 1
fi

echo "============================================="
echo "共检测到 ${TOTAL_COUNT} 个 PKL 文件，开始批量转换"
echo "输入目录：${PKL_FOLDER}"
echo "输出目录：${OUTPUT_DIR}"
echo "机器人型号：${ROBOT_TYPE}"
echo "============================================="

success=0
fail=0

# 循环转换
for pkl_path in "${PKL_LIST[@]}"; do
    # 提取纯文件名（去除路径、后缀）
    file_base=$(basename "${pkl_path}" .pkl)
    out_name="${file_base}_bvh"

    echo -e "\n【正在处理】源文件：${pkl_path}"
    echo "输出文件名：${out_name}.npz"

    # 执行转换命令（带headless无头模式）
    PYTHONPATH=. python scripts/gmr_to_npz_inter.py \
        --input_file "${pkl_path}" \
        --input_fps ${INPUT_FPS} \
        --frame_range ${FRAME_START} ${FRAME_END} \
        --output_name ${out_name} \
        --output_dir ${OUTPUT_DIR} \
        --output_fps ${OUTPUT_FPS} \
        --robot ${ROBOT_TYPE} \
        --start_frames ${START_FRAMES} \
        --end_frames ${END_FRAMES} \
        --hold_pos ${HOLD_POS} \
        --headless

    # 判断执行结果
    if [ $? -eq 0 ]; then
        echo "✅ ${file_base} 转换完成"
        success=$((success + 1))
    else
        echo "❌ ${file_base} 转换失败"
        fail=$((fail + 1))
    fi
done

# 汇总输出
echo -e "\n============================================="
echo "批量转换全部结束！"
echo "成功：${success} 个"
echo "失败：${fail} 个"
echo "NPZ文件存放路径：${OUTPUT_DIR}/"
echo "============================================="

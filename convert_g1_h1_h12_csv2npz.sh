#!/bin/bash
ROOT="/home/felix/isaac/xMimic"
CSV_ROOT="${ROOT}/motion_data/LAFAN1"
SCRIPT="${ROOT}/scripts/csv_to_npz.py"

# 子文件夹与robot_type一一对应
declare -A folder_robot_map=(
    ["g1"]="g1"
    ["h1"]="h1"
    ["h1_2"]="h1_2"
)

# 遍历LAFAN1下三个子文件夹
for SUB_FOLDER in g1 h1 h1_2; do
    ROBOT_TYPE=${folder_robot_map[$SUB_FOLDER]}
    INPUT_FOLDER="${CSV_ROOT}/${SUB_FOLDER}"
    # 输出目录：同级新建 {子文件夹}_npz 存放转换后的npz
    OUTPUT_FOLDER="${CSV_ROOT}/${SUB_FOLDER}_npz"
    # 不存在则自动创建输出文件夹
    mkdir -p "${OUTPUT_FOLDER}"

    # 遍历当前子文件夹全部csv
    for csv_file in "${INPUT_FOLDER}"/*.csv; do
        # 提取动作文件名（去除.csv后缀）
        motion_name=$(basename "${csv_file}" .csv)

        python "${SCRIPT}" \
            --input_file "${csv_file}" \
            --input_fps 30 \
            --output_name "${motion_name}" \
            --output_dir "${OUTPUT_FOLDER}" \
	    --robot_type "${ROBOT_TYPE}" \
            --headless
    done
done


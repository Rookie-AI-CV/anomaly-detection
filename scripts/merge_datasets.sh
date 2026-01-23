#!/bin/bash
dataset_root="/root/autodl-tmp/"
dirnames=(
    # "训练集+验证集（璧山13-14-19线）_V6"
    # "训练集+验证集（铜梁8l）_V20"
    # "训练集+验证集（通用）_V106"
    # "临时集（通用-铜梁）_V30"
    # "璧山19线误识别数据（反面天鹅颈）_V10"
    "训练集+验证集(毛肧未车光)_V2"
    # "训练集+验证集(轻微未车光屏蔽数据集)_V2"
    # "训练集+验证集(连接斜面未车光)_V4"
)

full_paths=()
for dirname in "${dirnames[@]}"; do
    clean_dirname=$(echo "$dirname" | xargs)
    full_paths+=("${dataset_root}${clean_dirname}")
done

echo "合并以下数据集:"
for path in "${full_paths[@]}"; do
    echo "  $path"
done

# 是否使用序号命名（设置为1启用，0或不设置则使用原始文件名）
USE_INDEXED_NAMES="1"

ARGS=(
    --datasets "${full_paths[@]}"
    --output_dir  /root/autodl-tmp/maopiweicheguang2
    --train_ratio 0.0
    --seed 42
)

if [ "$USE_INDEXED_NAMES" = "1" ]; then
    ARGS+=(--use-indexed-names)
fi

python3 scripts/merge_coco_datasets.py "${ARGS[@]}"

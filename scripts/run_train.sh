#!/bin/bash
cd "$(dirname "$0")/.."
python scripts/train.py \
    --config configs/dinov3_patchcore_style.yaml \
    --data-path /root/autodl-tmp/zhenwen_dataset/train/good \
    --model-path /root/autodl-tmp/model/vit_base_patch16_dinov3.lvd1689m/model.safetensors

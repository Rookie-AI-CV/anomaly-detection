#!/bin/bash
cd "$(dirname "$0")/.."
python scripts/train.py \
    --config configs/dinov3_image_level.yaml \
    --data-path /root/autodl-tmp/zhenwen_dataset/train/good \
    --model-path /root/anomaly-detection/outputs-classifier/best_model.pth 

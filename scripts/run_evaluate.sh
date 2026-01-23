# 评估
python scripts/evaluate.py \
    --checkpoint outputs/best_checkpoint.pth \
    --config configs/dinov3_image_level.yaml \
    --model-path /root/autodl-tmp/model/vit_base_patch16_dinov3.lvd1689m/model.safetensors \
    --test-data /root/autodl-tmp/maopiweicheguang2/test \
    --output new-base-out-mp \
    --find-threshold

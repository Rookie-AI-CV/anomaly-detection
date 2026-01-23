#!/usr/bin/env python3
"""检查训练配置"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hq_anomaly_detection.core.config import Config


def check_config(config_path: str):
    config = Config.from_file(config_path)
    
    print("=" * 50)
    print("训练配置检查")
    print("=" * 50)
    
    # GPU
    print("\n【GPU】")
    device = config.get("training.device", "cuda")
    device_id = config.get("training.device_id", 0)
    print(f"  设备: {device}:{device_id}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(device_id)}")
    
    # 数据
    print("\n【数据】")
    print(f"  路径: {config.get('data.train_data_path')}")
    print(f"  图像大小: {config.get('data.image_size', 224)}")
    print(f"  Batch: {config.get('data.batch_size', 32)}")
    
    # 模型
    print("\n【模型】")
    print(f"  DINOv3: {config.get('model.dino_model_name')}")
    print(f"  模式: {config.get('model.detection_mode', 'cls')}")
    print(f"  num_centers: {config.get('model.num_centers', 50000)}")
    print(f"  reduce_dim: {config.get('model.reduce_dim')}")
    
    # 输出
    print("\n【输出】")
    print(f"  目录: {config.get('output.dir', './outputs')}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dinov3_image_level.yaml")
    args = parser.parse_args()
    check_config(args.config)

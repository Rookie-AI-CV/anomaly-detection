#!/usr/bin/env python3
"""
检查训练配置：GPU、图像大小、batch_size 等信息
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from hq_anomaly_detection.core.config import Config

def check_config(config_path: str = "configs/dinov3_image_level.yaml"):
    """检查配置文件中的设置"""
    config = Config.from_file(config_path)
    
    print("=" * 60)
    print("训练配置检查")
    print("=" * 60)
    
    # GPU 信息
    print("\n【GPU 配置】")
    device_str = config.get("training.device", "cuda")
    device_id = config.get("training.device_id", 0)
    print(f"  配置的设备: {device_str}")
    print(f"  GPU ID: {device_id}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"  当前 CUDA 设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(device_id)}")
        if device_str == "cuda":
            actual_device = torch.device(f"cuda:{device_id}")
            print(f"  实际使用设备: {actual_device}")
        else:
            print(f"  实际使用设备: cpu (配置为 {device_str})")
    else:
        print(f"  实际使用设备: cpu (CUDA 不可用)")
    
    # 数据配置
    print("\n【数据配置】")
    image_size = config.get("data.image_size", 224)
    batch_size = config.get("data.batch_size", 32)
    num_workers = config.get("data.num_workers", 8)
    train_data_path = config.get("data.train_data_path", "./data/train")
    
    print(f"  图像大小: {image_size} x {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  训练数据路径: {train_data_path}")
    
    # 模型配置
    print("\n【模型配置】")
    model_name = config.get("model.name", "unknown")
    dino_model_name = config.get("model.dino_model_name", "unknown")
    model_path = config.get("model.model_path")
    num_centers = config.get("model.num_centers", 50000)
    buffer_size = config.get("model.buffer_size", 2500000)
    
    print(f"  模型类型: {model_name}")
    print(f"  DINOv3 模型: {dino_model_name}")
    print(f"  模型路径: {model_path if model_path else '自动下载'}")
    print(f"  内存银行中心数: {num_centers}")
    print(f"  缓冲区大小: {buffer_size}")
    
    # 输出配置
    print("\n【输出配置】")
    checkpoint_dir = config.get("output.checkpoint_dir", "./checkpoints")
    results_dir = config.get("output.results_dir", "./results")
    log_dir = config.get("output.log_dir", "./logs")
    
    print(f"  检查点目录: {checkpoint_dir}")
    print(f"  结果目录: {results_dir}")
    print(f"  日志目录: {log_dir}")
    
    print("\n" + "=" * 60)
    
    # 验证 GPU 使用
    if device_str == "cuda" and torch.cuda.is_available():
        print("\n✓ GPU 配置正确，训练将使用 GPU")
    elif device_str == "cuda" and not torch.cuda.is_available():
        print("\n⚠ 警告: 配置为使用 GPU，但 CUDA 不可用，将使用 CPU")
    else:
        print("\n✓ 配置为使用 CPU")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="检查训练配置")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dinov3_image_level.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    check_config(args.config)

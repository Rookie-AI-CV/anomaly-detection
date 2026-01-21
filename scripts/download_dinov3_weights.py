#!/usr/bin/env python3
"""
手动下载 DINOv3 预训练权重脚本

如果自动下载失败，可以使用此脚本手动下载权重文件。
"""

import os
import sys
from pathlib import Path

# 设置 Hugging Face 镜像源
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加脚本目录到路径
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# 加载环境变量
from env_config import load_env
load_env()

import timm
import torch
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dinov3_weights(model_name: str = "dinov3_base", output_dir: str = "./weights"):
    """
    下载 DINOv3 预训练权重
    
    Args:
        model_name: 模型名称 (dinov3_small, dinov3_base, dinov3_large)
        output_dir: 输出目录
    """
    # 模型名称映射
    model_map = {
        "dinov3_small": "vit_small_patch16_dinov3",
        "dinov3_base": "vit_base_patch16_dinov3",
        "dinov3_large": "vit_large_patch16_dinov3",
        "dinov3_huge": "vit_huge_patch16_dinov3",
    }
    
    # Hugging Face 仓库映射
    hf_repo_map = {
        "dinov3_small": "timm/vit_small_patch16_dinov3.lvd1689m",
        "dinov3_base": "timm/vit_base_patch16_dinov3.lvd1689m",
        "dinov3_large": "timm/vit_large_patch16_dinov3.lvd1689m",
        "dinov3_huge": "timm/vit_huge_patch16_dinov3.lvd1689m",
    }
    
    timm_model_name = model_map.get(model_name, model_name)
    hf_repo = hf_repo_map.get(model_name, f"timm/{timm_model_name}.lvd1689m")
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    hf_url = f"{hf_endpoint}/{hf_repo}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info(f"Downloading {timm_model_name} pretrained weights...")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Timm Model Name: {timm_model_name}")
    logger.info(f"Hugging Face Repository: {hf_repo}")
    logger.info(f"Hugging Face URL: {hf_url}")
    logger.info(f"HF_ENDPOINT: {hf_endpoint}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)
    
    try:
        # 方法1: 使用 timm 创建模型并保存权重
        logger.info("Method 1: Loading model via timm...")
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=0,
        )
        
        # 保存权重
        weight_path = output_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), weight_path)
        logger.info(f"✓ Successfully downloaded and saved weights to: {weight_path}")
        
        return str(weight_path)
        
    except Exception as e:
        logger.error(f"Method 1 failed: {e}")
        logger.info("Trying Method 2: Direct download from Hugging Face...")
        
        try:
            repo_id = f"timm/{timm_model_name}.lvd1689m"
            filename = "pytorch_model.bin"
            
            logger.info(f"Downloading from {repo_id}/{filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(output_dir),
            )
            
            weight_path = output_dir / f"{model_name}.pth"
            import shutil
            shutil.copy2(downloaded_path, weight_path)
            logger.info(f"✓ Successfully downloaded and saved weights to: {weight_path}")
            
            return str(weight_path)
            
        except Exception as e2:
            logger.error("=" * 80)
            logger.error("Both download methods failed!")
            logger.error("=" * 80)
            logger.error(f"Method 1 error: {e}")
            logger.error(f"Method 2 error: {e2}")
            logger.error("")
            logger.error("Download information:")
            logger.error(f"  - Hugging Face Repository: {hf_repo}")
            logger.error(f"  - Hugging Face URL: {hf_url}")
            logger.error(f"  - HF_ENDPOINT: {hf_endpoint}")
            logger.error("")
            logger.error("You can try:")
            logger.error("  1. Check your network connection")
            logger.error("  2. Visit the URL above and download manually")
            logger.error(f"  3. Use: huggingface-cli download {hf_repo} --local-dir {output_dir}")
            logger.error("=" * 80)
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DINOv3 pretrained weights")
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_base",
        choices=["dinov3_small", "dinov3_base", "dinov3_large", "dinov3_huge"],
        help="Model name to download (default: dinov3_base)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./weights",
        help="Output directory for weights (default: ./weights)"
    )
    
    args = parser.parse_args()
    
    try:
        weight_path = download_dinov3_weights(args.model_name, args.output_dir)
        print(f"\n✓ Download completed!")
        print(f"  Weight file: {weight_path}")
        print(f"\nYou can now use this weight file in your config:")
        print(f"  model_path: {weight_path}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)

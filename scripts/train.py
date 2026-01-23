#!/usr/bin/env python3
"""训练脚本"""

import argparse
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import common  
from hq_anomaly_detection import AnomalyDetector
from hq_anomaly_detection.core.config import Config


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    # 加载配置
    config = Config.from_file(args.config)
    if args.data_path:
        config.set("data.train_data_path", args.data_path)
    if args.model_path:
        config.set("model.model_path", args.model_path)
    
    # 输出目录
    output_dir = Path(config.get("output.dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志
    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {config.get('model.detection_mode', 'cls')}")
    logger.info(f"Output: {output_dir}")
    
    # 临时配置
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save(f.name)
        temp_config = f.name
    
    try:
        detector = AnomalyDetector(config_path=temp_config)
        detector.train()
        
        checkpoint_path = output_dir / "best_checkpoint.pth"
        detector.save_model(checkpoint_path)
        
        # 打印结果
        import torch
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        if 'cls_level' in ckpt and ckpt['cls_level'].get('memory_bank') is not None:
            shape = ckpt['cls_level']['memory_bank'].shape
            logger.info(f"CLS memory bank: {shape[0]} x {shape[1]}")
        
        if 'patch_level' in ckpt and ckpt['patch_level'].get('memory_bank') is not None:
            shape = ckpt['patch_level']['memory_bank'].shape
            logger.info(f"Patch memory bank: {shape[0]} x {shape[1]}")
        
        logger.info("=" * 50)
        
    finally:
        Path(temp_config).unlink()


if __name__ == "__main__":
    main()

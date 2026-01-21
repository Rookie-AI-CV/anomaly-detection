#!/usr/bin/env python3
"""
DINOv3 image level anomaly detection training script.
"""

import os
import argparse
import sys
import tempfile
import logging
from pathlib import Path
from datetime import datetime

# Set Hugging Face mirror endpoint (use environment variable if set, otherwise use mirror)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add scripts directory to path for importing env_config
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Load environment variables before importing other modules
from env_config import load_env
load_env()

from hq_anomaly_detection import AnomalyDetector
from hq_anomaly_detection.core.config import Config


def main():
    parser = argparse.ArgumentParser(description="DINOv3 image level anomaly detection training")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Training data path (overrides config file)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model path (overrides config file, if provided, don't download automatically)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dinov3_image_level.yaml",
        help="Config file path (default: configs/dinov3_image_level.yaml)"
    )
    
    args = parser.parse_args()
    
    # Config file path
    config_path = args.config
    
    print("=" * 50)
    print("DINOv3 image level anomaly detection training")
    print("=" * 50)
    print(f"config file: {config_path}")
    if args.data_path:
        print(f"data path: {args.data_path} (from command line)")
    if args.model_path:
        print(f"model path: {args.model_path} (from command line)")
    print("=" * 50)
    
    # load config
    config = Config.from_file(config_path)
    
    # override config with command line arguments
    if args.data_path:
        config.set("data.train_data_path", args.data_path)
    
    if args.model_path:
        config.set("model.model_path", args.model_path)
    
    # create temporary config file with overridden values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save(f.name)
        temp_config_path = f.name
    
    try:
        # create output directory
        checkpoint_dir = Path(config.get("output.checkpoint_dir"))
        results_dir = Path(config.get("output.results_dir"))
        log_dir = Path(config.get("output.log_dir"))
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging configuration
        log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("DINOv3 Image Level Anomaly Detection Training")
        logger.info("=" * 60)
        logger.info(f"Config file: {config_path}")
        logger.info(f"Temporary config file: {temp_config_path}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(f"Results directory: {results_dir}")
        logger.info(f"Log directory: {log_dir}")
        if args.data_path:
            logger.info(f"Data path: {args.data_path} (from command line)")
        if args.model_path:
            logger.info(f"Model path: {args.model_path} (from command line)")
        logger.info("=" * 60)
        
        # create detector (automatically load all parameters from config file)
        logger.info("Initializing detector...")
        detector = AnomalyDetector(config_path=temp_config_path)
        logger.info("Detector initialized successfully")
        
        # train (read all parameters from config file)
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)
        detector.train()
        
        # save model
        checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
        logger.info("=" * 60)
        logger.info(f"Saving model to: {checkpoint_path}")
        logger.info("=" * 60)
        detector.save_model(checkpoint_path)
        
        # Print saved information
        import torch
        checkpoint_info = torch.load(checkpoint_path, map_location='cpu')
        logger.info("=" * 60)
        logger.info("Checkpoint saved successfully!")
        logger.info("=" * 60)
        logger.info(f"Checkpoint file: {checkpoint_path}")
        logger.info(f"File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
        if 'memory_bank' in checkpoint_info:
            mb_shape = checkpoint_info['memory_bank'].shape
            logger.info(f"Memory bank shape: {mb_shape}")
            logger.info(f"Memory bank size: {mb_shape[0]} samples × {mb_shape[1]} dimensions")
        logger.info(f"Embedding dimension: {checkpoint_info.get('embed_dim', 'N/A')}")
        logger.info(f"Number of centers: {checkpoint_info.get('num_centers', 'N/A')}")
        logger.info(f"Model name: {checkpoint_info.get('model_name', 'N/A')}")
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("Training Completed!")
        print("=" * 60)
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        print(f"✓ Log file: {log_file}")
        print(f"✓ Memory bank saved with {checkpoint_info.get('num_centers', 'N/A')} feature vectors")
        print("=" * 60)
    finally:
        # cleanup temporary config file
        Path(temp_config_path).unlink()


if __name__ == "__main__":
    main()

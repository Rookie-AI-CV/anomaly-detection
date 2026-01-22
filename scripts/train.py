#!/usr/bin/env python3
"""
Unified training script for anomaly detection models.

Supports:
- DINOv3 image-level detection (cls_token)
- DINOv3 PatchCore-style detection (patch tokens)
- Automatically selects training mode based on config file
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
    parser = argparse.ArgumentParser(
        description="Unified anomaly detection training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train.py --config configs/dinov3_image_level.yaml
  
  # Train with custom data path
  python scripts/train.py --config configs/dinov3_patchcore_style.yaml --data-path ./data/train
  
  # Train with custom model path
  python scripts/train.py --config configs/dinov3_image_level.yaml --model-path ./weights/dinov3_base.pth
        """
    )
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
        required=True,
        help="Config file path (e.g., configs/dinov3_image_level.yaml or configs/dinov3_patchcore_style.yaml)"
    )
    
    args = parser.parse_args()
    
    # Config file path
    config_path = args.config
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = Config.from_file(config_path)
    
    # Get model name and detection mode
    model_name = config.get("model.name")
    detection_mode = config.get("model.detection_mode", "image_level")
    
    print("=" * 60)
    print("Anomaly Detection Training")
    print("=" * 60)
    print(f"Config file: {config_path}")
    print(f"Model: {model_name}")
    print(f"Detection mode: {detection_mode}")
    if args.data_path:
        print(f"Data path: {args.data_path} (from command line)")
    if args.model_path:
        print(f"Model path: {args.model_path} (from command line)")
    print("=" * 60)
    
    # Override config with command line arguments
    if args.data_path:
        config.set("data.train_data_path", args.data_path)
    
    if args.model_path:
        config.set("model.model_path", args.model_path)
    
    # Create temporary config file with overridden values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save(f.name)
        temp_config_path = f.name
    
    try:
        # Create output directory
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
        logger.info("Anomaly Detection Training")
        logger.info("=" * 60)
        logger.info(f"Config file: {config_path}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Detection mode: {detection_mode}")
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
        
        # Create detector (automatically load all parameters from config file)
        logger.info("Initializing detector...")
        detector = AnomalyDetector(config_path=temp_config_path)
        logger.info("Detector initialized successfully")
        
        # Train (read all parameters from config file)
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)
        detector.train()
        
        # Save model
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
        
        # Print memory bank info
        if 'memory_bank' in checkpoint_info:
            mb_shape = checkpoint_info['memory_bank'].shape
            logger.info(f"Image-level memory bank shape: {mb_shape}")
            logger.info(f"Image-level memory bank size: {mb_shape[0]} samples × {mb_shape[1]} dimensions")
        
        if 'patch_memory_bank' in checkpoint_info:
            pmb_shape = checkpoint_info['patch_memory_bank'].shape
            logger.info(f"Patch-level memory bank shape: {pmb_shape}")
            logger.info(f"Patch-level memory bank size: {pmb_shape[0]} samples × {pmb_shape[1]} dimensions")
        
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
        print(f"✓ Detection mode: {detection_mode}")
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        print(f"✓ Log file: {log_file}")
        if 'memory_bank' in checkpoint_info:
            print(f"✓ Image-level memory bank: {checkpoint_info['memory_bank'].shape[0]} feature vectors")
        if 'patch_memory_bank' in checkpoint_info:
            print(f"✓ Patch-level memory bank: {checkpoint_info['patch_memory_bank'].shape[0]} patch feature vectors")
        print("=" * 60)
    finally:
        # Cleanup temporary config file
        Path(temp_config_path).unlink()


if __name__ == "__main__":
    main()

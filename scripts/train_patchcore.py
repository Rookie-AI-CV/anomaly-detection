#!/usr/bin/env python3
"""
PatchCore anomaly detection training script.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add scripts directory to path for importing env_config
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Load environment variables before importing other modules
from env_config import load_env
load_env()

# Import training function from hq_anomaly_detection
from hq_anomaly_detection.models.patchcore import train_patchcore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PatchCore anomaly detection training")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Training data path (overrides config file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (optional)"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PatchCore anomaly detection training")
    print("=" * 50)
    
    # Get data path from command line or config
    if args.data_path:
        data_path = args.data_path
        print(f"data path: {data_path} (from command line)")
    else:
        # Try to get from config file
        if args.config:
            try:
                from hq_anomaly_detection.core.config import Config
                config = Config.from_file(args.config)
                data_path = config.get("data.train_data_path")
                if data_path:
                    print(f"data path: {data_path} (from config file)")
                else:
                    raise ValueError("Config file does not contain 'data.train_data_path'")
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load config file: {args.config}. "
                    f"Please provide --data-path or ensure config file exists. Error: {e}"
                )
        else:
            raise ValueError(
                "Please provide either --data-path or --config argument"
            )
    
    print(f"device ID: {args.device_id}")
    if args.config:
        print(f"config file: {args.config}")
    print("=" * 50)
    
    # Check if data path exists
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Run training
    try:
        train_patchcore(
            data_path=data_path,
            config_path=args.config,
            device_id=args.device_id,
        )
        print("=" * 50)
        print("Training completed!")
        print("=" * 50)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print("=" * 50)
        print(f"Training failed: {e}")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import argparse
import sys
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from env_config import load_env
load_env()

from hq_anomaly_detection import AnomalyDetector
from hq_anomaly_detection.core.config import Config


def get_image_paths(input_path: str) -> List[Path]:
    input_path = Path(input_path)
    image_paths = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            image_paths.append(input_path)
        else:
            raise ValueError(f"Unsupported format: {input_path.suffix}")
    elif input_path.is_dir():
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            image_paths.extend(input_path.rglob(f"*{ext}"))
            image_paths.extend(input_path.rglob(f"*{ext.upper()}"))
        image_paths = sorted(image_paths)
    else:
        raise ValueError(f"Path not found: {input_path}")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found: {input_path}")
    
    return image_paths


def save_results(results: List[Dict], output_dir: Path, save_json: bool, save_csv: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_json:
        json_path = output_dir / f"results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON: {json_path}")
    
    if save_csv:
        csv_path = output_dir / f"results_{timestamp}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'anomaly_score', 'prediction'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'image_path': r['image_path'],
                    'anomaly_score': f"{r['anomaly_score']:.6f}",
                    'prediction': 'anomaly' if r['prediction'] else 'normal'
                })
        print(f"CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/dinov3_image_level.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    image_paths = get_image_paths(args.image)
    print(f"Images: {len(image_paths)}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config.from_file(config_path)
    config.set("training.device_id", args.device_id)
    
    detector = AnomalyDetector(config_path=config_path, checkpoint_path=checkpoint_path)
    detector.load_model(checkpoint_path)
    
    results = []
    if len(image_paths) > 1:
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i + args.batch_size]
            batch_results = detector.predict_batch(
                [str(p) for p in batch_paths],
                threshold=args.threshold
            )
            for img_path, result in zip(batch_paths, batch_results):
                results.append({
                    'image_path': str(img_path),
                    'anomaly_score': result['anomaly_score'],
                    'prediction': result['prediction']
                })
            print(f"Processed: {min(i + args.batch_size, len(image_paths))}/{len(image_paths)}")
    else:
        result = detector.detect(str(image_paths[0]), threshold=args.threshold)
        results.append({
            'image_path': str(image_paths[0]),
            'anomaly_score': result['anomaly_score'],
            'prediction': result['prediction']
        })
    
    anomaly_count = sum(1 for r in results if r['prediction'])
    normal_count = len(results) - anomaly_count
    avg_score = np.mean([r['anomaly_score'] for r in results])
    
    print(f"\nNormal: {normal_count}, Anomaly: {anomaly_count}")
    print(f"Avg score: {avg_score:.6f}")
    
    if args.save_json or args.save_csv:
        save_results(results, output_dir, args.save_json, args.save_csv)
    else:
        save_results(results, output_dir, True, False)


if __name__ == "__main__":
    main()

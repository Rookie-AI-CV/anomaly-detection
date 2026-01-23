#!/usr/bin/env python3
"""推理脚本"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np

import common  
from hq_anomaly_detection import AnomalyDetector


def get_image_paths(input_path: str) -> List[Path]:
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return [input_path]
        raise ValueError(f"Unsupported format: {input_path.suffix}")
    
    if input_path.is_dir():
        paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
            paths.extend(input_path.rglob(f"*.{ext}"))
            paths.extend(input_path.rglob(f"*.{ext.upper()}"))
        if not paths:
            raise ValueError(f"No images found: {input_path}")
        return sorted(paths)
    
    raise ValueError(f"Path not found: {input_path}")


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
                    'prediction': 'anomaly' if r.get('prediction') else 'normal'
                })
        print(f"CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/dinov3_image_level.yaml")
    parser.add_argument("--model-path", type=str, help="Model weights path")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    config = Config.from_file(args.config)
    if args.model_path:
        config.set("model.model_path", args.model_path)
    
    image_paths = get_image_paths(args.image)
    print(f"Images: {len(image_paths)}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建检测器（构造函数会自动加载 checkpoint）
    detector = AnomalyDetector(config=config, checkpoint_path=args.checkpoint)
    
    # 推理
    results = []
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_results = detector.predict_batch([str(p) for p in batch_paths], threshold=args.threshold)
        
        for img_path, result in zip(batch_paths, batch_results):
            results.append({
                'image_path': str(img_path),
                'anomaly_score': result['anomaly_score'],
                'prediction': result.get('prediction')
            })
        
        print(f"Processed: {min(i + args.batch_size, len(image_paths))}/{len(image_paths)}")
    
    # 统计
    anomaly_count = sum(1 for r in results if r.get('prediction'))
    avg_score = np.mean([r['anomaly_score'] for r in results])
    print(f"\nNormal: {len(results) - anomaly_count}, Anomaly: {anomaly_count}")
    print(f"Avg score: {avg_score:.6f}")
    
    # 保存
    save_json = args.save_json or (not args.save_json and not args.save_csv)
    save_results(results, output_dir, save_json, args.save_csv)


if __name__ == "__main__":
    main()

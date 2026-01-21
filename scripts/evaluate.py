#!/usr/bin/env python3
import os
import argparse
import sys
import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import numpy as np

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from env_config import load_env
load_env()

from hq_anomaly_detection import AnomalyDetector
from hq_anomaly_detection.core.config import Config
from hq_anomaly_detection.utils.data_loader import load_dataset


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) == 2 else None
    except ValueError:
        auc_roc = None
    
    # Calculate confusion matrix with explicit labels to ensure 2x2 shape
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc) if auc_roc is not None else None,
        'specificity': float(specificity),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, Dict]:
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = calculate_metrics(y_true, y_pred, y_scores)
            best_metrics['threshold'] = float(threshold)
    
    return best_threshold, best_metrics


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: Path):
    try:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        if len(np.unique(y_true)) < 2:
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC: {save_path}")
    except ImportError:
        pass


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CM: {save_path}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/dinov3_image_level.yaml")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--find-threshold", action="store_true")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    if args.threshold is None and not args.find_threshold:
        raise ValueError("Need --threshold or --find-threshold")
    
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    
    if not checkpoint_path.exists() or not config_path.exists():
        raise FileNotFoundError("Path not found")
    
    config = Config.from_file(config_path)
    data_format = config.get("data.data_format", "custom_mvtec")
    
    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        train_data_path = Path(config.get("data.train_data_path", "./data/train"))
        test_data_path = train_data_path.parent / "test"
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "mvtec" in data_format:
        data_root = test_data_path.parent
        load_format = "mvtec"
        split_name = test_data_path.name
    else:
        data_root = test_data_path.parent
        load_format = "custom"
        split_name = test_data_path.name
    
    test_data = load_dataset(
        str(data_root),
        data_format=load_format,
        split=split_name
    )
    
    if len(test_data) == 0:
        if load_format == "mvtec":
            normal_dir = data_root / split_name / "good"
            print(f"Looking for data in: {normal_dir}")
            if split_name == "test":
                print(f"Looking for anomalies in: {data_root / split_name}")
        else:
            normal_dir = data_root / split_name / "normal"
            anomaly_dir = data_root / split_name / "anomaly"
            print(f"Looking for data in: {normal_dir}, {anomaly_dir}")
        raise ValueError(f"No test data found in: {test_data_path} (data_root={data_root}, split={split_name}, format={load_format})")
    
    image_paths = [Path(path) for path, _ in test_data]
    labels = np.array([label for _, label in test_data])
    
    config.set("training.device_id", args.device_id)
    
    detector = AnomalyDetector(config_path=config_path, checkpoint_path=checkpoint_path)
    detector.load_model(checkpoint_path)
    
    y_scores = []
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_results = detector.predict_batch([str(p) for p in batch_paths], threshold=None)
        y_scores.extend([r['anomaly_score'] for r in batch_results])
        print(f"Processed: {min(i + args.batch_size, len(image_paths))}/{len(image_paths)}")
    
    y_scores = np.array(y_scores)
    
    if args.find_threshold:
        threshold, metrics = find_optimal_threshold(labels, y_scores)
        print(f"Optimal threshold: {threshold:.6f} (F1={metrics['f1_score']:.4f})")
        y_pred = (y_scores >= threshold).astype(int)
    else:
        threshold = args.threshold
        y_pred = (y_scores >= threshold).astype(int)
        metrics = calculate_metrics(labels, y_pred, y_scores)
        metrics['threshold'] = float(threshold)
    
    print(f"\nThreshold: {metrics['threshold']:.6f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1_score']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, TP: {cm['tp']}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"eval_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'threshold': metrics['threshold'],
            'metrics': metrics,
            'score_stats': {
                'mean': float(np.mean(y_scores)),
                'std': float(np.std(y_scores)),
                'min': float(np.min(y_scores)),
                'max': float(np.max(y_scores))
            }
        }, f, indent=2)
    print(f"\nResults: {json_path}")
    
    csv_path = output_dir / f"eval_{timestamp}.csv"
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'anomaly_score', 'is_correct'])
        for img_path, true_label, pred_label, score in zip(image_paths, labels, y_pred, y_scores):
            writer.writerow([str(img_path), int(true_label), int(pred_label), float(score), int(true_label == pred_label)])
    print(f"CSV: {csv_path}")
    
    if args.plot:
        y_pred = (y_scores >= threshold).astype(int)
        if len(np.unique(labels)) == 2:
            plot_roc_curve(labels, y_scores, output_dir / f"roc_{timestamp}.png")
        plot_confusion_matrix(labels, y_pred, output_dir / f"cm_{timestamp}.png")


if __name__ == "__main__":
    main()

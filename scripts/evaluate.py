#!/usr/bin/env python3
"""评估脚本"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np

import common  
from hq_anomaly_detection import AnomalyDetector
from hq_anomaly_detection.core.config import Config
from hq_anomaly_detection.utils.data_loader import load_dataset


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    
    try:
        if len(np.unique(y_true)) == 2:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        pass
    
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray):
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 1000)
    best_threshold, best_f1 = 0.5, 0.0
    
    for t in thresholds:
        f1 = f1_score(y_true, (y_scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    
    y_pred = (y_scores >= best_threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    metrics['threshold'] = float(best_threshold)
    return best_threshold, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate script")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/dinov3_image_level.yaml")
    parser.add_argument("--model-path", type=str, help="Model weights path")
    parser.add_argument("--test-data", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--find-threshold", action="store_true")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    if args.threshold is None and not args.find_threshold:
        raise ValueError("Need --threshold or --find-threshold")
    
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    config = Config.from_file(args.config)
    if args.model_path:
        config.set("model.model_path", args.model_path)
    data_format = config.get("data.data_format", "custom_mvtec")
    
    # 测试数据路径
    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        train_path = Path(config.get("data.train_data_path", "./data/train"))
        test_data_path = train_path.parent / "test"
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    data_root = test_data_path.parent
    load_format = "mvtec" if "mvtec" in data_format else "custom"
    test_data = load_dataset(str(data_root), data_format=load_format, split=test_data_path.name)
    
    if len(test_data) == 0:
        raise ValueError(f"No test data found: {test_data_path}")
    
    image_paths = [Path(path) for path, _ in test_data]
    labels = np.array([label for _, label in test_data])
    
    # 创建检测器
    detector = AnomalyDetector(config=config, checkpoint_path=args.checkpoint)
    
    # 推理
    y_scores = []
    for i in range(0, len(image_paths), args.batch_size):
        batch = image_paths[i:i + args.batch_size]
        results = detector.predict_batch([str(p) for p in batch])
        y_scores.extend([r['anomaly_score'] for r in results])
        print(f"Processed: {min(i + args.batch_size, len(image_paths))}/{len(image_paths)}")
    
    y_scores = np.array(y_scores)
    
    # 计算指标
    if args.find_threshold:
        threshold, metrics = find_optimal_threshold(labels, y_scores)
        print(f"Optimal threshold: {threshold:.6f}")
    else:
        threshold = args.threshold
        y_pred = (y_scores >= threshold).astype(int)
        metrics = calculate_metrics(labels, y_pred, y_scores)
        metrics['threshold'] = float(threshold)
    
    # 打印结果
    print(f"\nThreshold: {metrics['threshold']:.6f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1_score']:.4f}")
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}, TP: {cm['tp']}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"eval_{timestamp}.json", 'w') as f:
        json.dump({'threshold': metrics['threshold'], 'metrics': metrics}, f, indent=2)
    
    y_pred = (y_scores >= threshold).astype(int)
    with open(output_dir / f"eval_{timestamp}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'anomaly_score', 'is_correct'])
        for path, true, pred, score in zip(image_paths, labels, y_pred, y_scores):
            writer.writerow([str(path), int(true), int(pred), float(score), int(true == pred)])
    
    print(f"\nResults saved to: {output_dir}")
    
    # 绘图
    if args.plot:
        try:
            from sklearn.metrics import roc_curve, auc, confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if len(np.unique(labels)) == 2:
                fpr, tpr, _ = roc_curve(labels, y_scores)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={auc(fpr, tpr):.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.legend()
                plt.savefig(output_dir / f"roc_{timestamp}.png", dpi=300)
                plt.close()
            
            cm = confusion_matrix(labels, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(output_dir / f"cm_{timestamp}.png", dpi=300)
            plt.close()
            
            print(f"Plots saved")
        except ImportError:
            print("matplotlib/seaborn not available for plotting")


if __name__ == "__main__":
    main()

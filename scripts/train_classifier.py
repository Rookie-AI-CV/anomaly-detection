#!/usr/bin/env python3
"""DINOv3分类模型训练脚本"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

import common
from hq_anomaly_detection.models.dinov3.feature_extractor import DINOv3FeatureExtractor
from hq_anomaly_detection.core.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClassificationDataset(Dataset):
    """按类别文件夹组织的分类数据集"""
    
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # 收集类别和图像
        class_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.samples.extend([(str(p), idx) for p in class_dir.glob(ext)])
                self.samples.extend([(str(p), idx) for p in class_dir.glob(ext.upper())])
        
        logger.info(f"Found {len(self.class_to_idx)} classes, {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DINOv3Classifier(nn.Module):
    """DINOv3分类模型"""
    
    def __init__(self, model_name, model_path, num_classes, unfreeze_patterns=None):
        super().__init__()
        self.feature_extractor = DINOv3FeatureExtractor(
            model_name=model_name, model_path=model_path, use_cls_token=True
        )
        self.classifier = nn.Linear(self.feature_extractor.embed_dim, num_classes)
        self._unfreeze_by_patterns(unfreeze_patterns or [])
    
    def _unfreeze_by_patterns(self, patterns):
        """通过正则表达式解冻层"""
        for param in self.feature_extractor.backbone_model.parameters():
            param.requires_grad = False
        
        if not patterns:
            logger.info("No unfreeze patterns specified, all backbone parameters are frozen")
            return
        
        all_names = [name for name, _ in self.feature_extractor.backbone_model.named_parameters()]
        logger.info(f"Sample layer names (first 5): {all_names[:5]}")
        
        # 解冻匹配正则表达式的层
        unfrozen_count = 0
        unfrozen_names = []
        for name, param in self.feature_extractor.backbone_model.named_parameters():
            for pattern in patterns:
                if re.search(pattern, name):
                    param.requires_grad = True
                    unfrozen_count += 1
                    unfrozen_names.append(name)
                    logger.info(f"Unfrozen: {name} (matched pattern: {pattern})")
                    break
        
        if unfrozen_count == 0:
            logger.warning(f"No layers matched patterns: {patterns}")
            logger.info("Available layer name patterns (first 20):")
            for name in all_names[:20]:
                logger.info(f"  {name}")
        
        logger.info(f"Unfrozen {unfrozen_count} parameters based on {len(patterns)} pattern(s)")
    
    def forward(self, x):
        features = self.feature_extractor.backbone_model.forward_features(x)
        if isinstance(features, dict):
            cls_token = (features.get('x_norm_clstoken') or 
                       features.get('cls_token') or
                       features.get('x_cls_token'))
            if cls_token is None:
                patch_tokens = features.get('x_norm_patchtokens') or features.get('patchtokens')
                cls_token = patch_tokens.mean(dim=1) if patch_tokens is not None else None
        else:
            cls_token = features[:, 0] if features.dim() == 3 else features
        return self.classifier(cls_token)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        _, predicted = torch.max(model(images).data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return loss_sum / len(dataloader), 100 * correct / total


def validate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            loss_sum += criterion(logits, labels).item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    acc = 100 * correct / total
    if return_predictions:
        return loss_sum / len(dataloader), acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    return loss_sum / len(dataloader), acc


def compute_metrics(y_true, y_pred, num_classes):
    """计算精确率、召回率、F1"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return precision, recall, f1, precision_macro, recall_macro, f1_macro


def plot_pr_curve(y_true, y_probs, num_classes, output_path):
    """绘制PR曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if num_classes == 2:
        # 二分类：使用正类
        y_binary = (y_true == 1).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, 1])
        ap = average_precision_score(y_binary, y_probs[:, 1])
        ax.plot(recall, precision, label=f'AP={ap:.3f}')
    else:
        # 多分类：每个类别一条曲线
        aps = []
        for i in range(num_classes):
            y_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, i])
            ap = average_precision_score(y_binary, y_probs[:, i])
            aps.append(ap)
            ax.plot(recall, precision, label=f'Class {i} (AP={ap:.3f})', alpha=0.7)
        
        # 宏平均
        ap_macro = np.mean(aps)
        ax.axhline(y=ap_macro, color='k', linestyle='--', label=f'Macro AP={ap_macro:.3f}')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--val-data-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--lr-classifier", type=float, default=None)
    parser.add_argument("--device-id", type=int, default=None)
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = Config.from_file(args.config)
    else:
        default_config = Path(__file__).parent.parent / "configs" / "classifier.yaml"
        if default_config.exists():
            config = Config.from_file(default_config)
            logger.info(f"Using default config: {default_config}")
        else:
            raise FileNotFoundError(f"Config file not found. Please specify --config or create {default_config}")
    
    # 从配置文件读取参数，命令行参数可覆盖
    train_data_path = args.train_data_path or config.get("data.train_data_path")
    val_data_path = args.val_data_path or config.get("data.val_data_path")
    model_name = args.model_name or config.get("model.name", "dinov3_base")
    model_path = args.model_path if args.model_path is not None else config.get("model.model_path")
    output_dir = Path(args.output_dir or config.get("output.dir", "./outputs-classifier"))
    batch_size = args.batch_size or config.get("data.batch_size", 32)
    epochs = args.epochs or config.get("training.epochs", 10)
    lr_backbone = args.lr_backbone if args.lr_backbone is not None else float(config.get("training.lr_backbone", 1e-5))
    lr_classifier = args.lr_classifier if args.lr_classifier is not None else float(config.get("training.lr_classifier", 1e-3))
    unfreeze_patterns = config.get("model.unfreeze_patterns", [])
    device_id = args.device_id if args.device_id is not None else config.get("training.device_id", 0)
    num_workers = config.get("data.num_workers", 4)
    
    if not train_data_path or not val_data_path:
        raise ValueError("train_data_path and val_data_path must be specified in config or via --train-data-path and --val-data-path")
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Config: {args.config or 'default'}")
    logger.info(f"Train data: {train_data_path}")
    logger.info(f"Val data: {val_data_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Unfreeze patterns: {unfreeze_patterns}")
    
    # 数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 分别加载训练集和验证集
    train_dataset = ClassificationDataset(train_data_path, transform=transform)
    val_dataset = ClassificationDataset(val_data_path, transform=transform)
    
    # 确保类别映射一致
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        logger.warning("Class mappings differ between train and val sets, using train set mapping")
    
    num_classes = len(train_dataset.class_to_idx)
    logger.info(f"Number of classes: {num_classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 模型
    model = DINOv3Classifier(model_name, model_path, num_classes, unfreeze_patterns).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 优化器（不同学习率）
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': classifier_params, 'lr': lr_classifier}
    ])
    
    # 训练
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 计算详细指标
            _, _, y_true, y_pred, y_probs = validate(model, val_loader, criterion, device, return_predictions=True)
            precision, recall, f1, prec_macro, rec_macro, f1_macro = compute_metrics(y_true, y_pred, num_classes)
            
            logger.info(f"\nMetrics (Macro): Precision={prec_macro:.4f}, Recall={rec_macro:.4f}, F1={f1_macro:.4f}")
            if num_classes <= 10:  # 只打印前10个类别的详细指标
                for i in range(num_classes):
                    logger.info(f"  Class {i}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
            
            # 绘制PR曲线
            pr_path = output_dir / f"pr_curve_epoch{epoch}.png"
            plot_pr_curve(y_true, y_probs, num_classes, pr_path)
            logger.info(f"PR curve saved to {pr_path}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_classes': num_classes,
                'class_to_idx': train_dataset.class_to_idx,
            }, output_dir / "best_model.pth")
            logger.info(f"Saved best model (val_acc={val_acc:.2f}%)")
    
    # 最终评估
    logger.info(f"\nFinal evaluation:")
    _, _, y_true, y_pred, y_probs = validate(model, val_loader, criterion, device, return_predictions=True)
    precision, recall, f1, prec_macro, rec_macro, f1_macro = compute_metrics(y_true, y_pred, num_classes)
    logger.info(f"Final Metrics (Macro): Precision={prec_macro:.4f}, Recall={rec_macro:.4f}, F1={f1_macro:.4f}")
    plot_pr_curve(y_true, y_probs, num_classes, output_dir / "pr_curve_final.png")
    logger.info(f"Final PR curve saved to {output_dir / 'pr_curve_final.png'}")
    logger.info(f"\nTraining completed! Best val_acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

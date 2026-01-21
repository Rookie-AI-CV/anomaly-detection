"""
数据加载工具

用于加载和预处理数据集。
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np


def load_dataset(
    data_root: str,
    data_format: str = "mvtec",
    split: str = "train"
) -> List[Tuple[str, int]]:
    """
    加载数据集
    
    Args:
        data_root: 数据根目录
        data_format: 数据格式 ("mvtec", "custom")
        split: 数据集划分 ("train", "test")
        
    Returns:
        (图像路径, 标签) 列表，标签: 0=正常, 1=异常
    """
    data_root = Path(data_root)
    data_list = []
    
    if data_format == "mvtec":
        # MVTec AD 格式
        normal_dir = data_root / split / "good"
        anomaly_dir = data_root / split
        
        # 正常样本
        if normal_dir.exists():
            for img_path in normal_dir.glob("*.png"):
                data_list.append((str(img_path), 0))
        
        # 异常样本（测试集）
        if split == "test":
            for anomaly_type_dir in anomaly_dir.iterdir():
                if anomaly_type_dir.is_dir() and anomaly_type_dir.name != "good":
                    for img_path in anomaly_type_dir.glob("*.png"):
                        data_list.append((str(img_path), 1))
    
    elif data_format == "custom":
        # 自定义格式
        split_dir = data_root / split
        if split_dir.exists():
            normal_dir = split_dir / "normal"
            anomaly_dir = split_dir / "anomaly"
            
            if normal_dir.exists():
                for img_path in normal_dir.glob("*.png"):
                    data_list.append((str(img_path), 0))
            
            if anomaly_dir.exists():
                for img_path in anomaly_dir.glob("*.png"):
                    data_list.append((str(img_path), 1))
    
    return data_list


def load_image(image_path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        size: 目标尺寸 (H, W)，可选
        
    Returns:
        图像数组 (H, W, C)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if size:
        image = cv2.resize(image, size[::-1])
    
    return image

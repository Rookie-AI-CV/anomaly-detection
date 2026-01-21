"""
可视化工具

用于可视化异常检测结果。
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import cv2


def visualize_detection_result(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    anomaly_score: float,
    threshold: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False
):
    """
    可视化异常检测结果
    
    Args:
        image: 原始图像 (H, W, C)
        anomaly_map: 异常热图 (H, W)
        anomaly_score: 异常分数
        threshold: 异常阈值（可选）
        save_path: 保存路径（可选）
        show: 是否显示图像
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 异常热图
    im = axes[1].imshow(anomaly_map, cmap='jet')
    axes[1].set_title(f"Anomaly Map (Score: {anomaly_score:.4f})")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 叠加结果
    overlay = cv2.addWeighted(
        image.astype(np.uint8),
        0.6,
        cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET),
        0.4,
        0
    )
    axes[2].imshow(overlay)
    title = f"Overlay (Anomaly: {anomaly_score > threshold if threshold else 'N/A'})"
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

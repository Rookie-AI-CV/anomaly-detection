"""
模型模块

包含各种异常检测模型的实现和封装。

结构：
- base/: 基础组件（如内存银行）
- dinov3/: DINOv3 相关模型
- patchcore/: PatchCore 相关模型（未来扩展）
"""

from hq_anomaly_detection.models.base import KCenterGreedyMemoryBank
from hq_anomaly_detection.models.dinov3 import (
    DINOv3FeatureExtractor,
    DINOv3ImageLevelDetector,
)

__all__ = [
    "KCenterGreedyMemoryBank",
    "DINOv3FeatureExtractor",
    "DINOv3ImageLevelDetector",
]

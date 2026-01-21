"""
基础模型组件

包含通用的基础类，如内存银行、特征提取器基类等。
"""

from hq_anomaly_detection.models.base.memory_bank import KCenterGreedyMemoryBank

__all__ = ["KCenterGreedyMemoryBank"]

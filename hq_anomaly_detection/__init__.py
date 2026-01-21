"""
HQ Anomaly Detection - 工业缺陷检测异常检测算法库

本库基于 anomalib 提供多种异常检测算法。
"""

__version__ = "0.1.0"
__author__ = "HQ Team"

from hq_anomaly_detection.core.detector import AnomalyDetector
from hq_anomaly_detection.core.config import Config

__all__ = [
    "AnomalyDetector",
    "Config",
]

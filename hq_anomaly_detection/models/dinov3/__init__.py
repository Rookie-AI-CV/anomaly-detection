"""
DINOv3 model module.

Contains DINOv3-based anomaly detection models.
"""

from hq_anomaly_detection.models.dinov3.feature_extractor import DINOv3FeatureExtractor
from hq_anomaly_detection.models.dinov3.image_level import DINOv3ImageLevelDetector

__all__ = [
    "DINOv3FeatureExtractor",
    "DINOv3ImageLevelDetector",
]

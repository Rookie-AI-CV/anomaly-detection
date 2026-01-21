"""
测试异常检测器
"""

import pytest
from pathlib import Path
from hq_anomaly_detection import AnomalyDetector


def test_detector_initialization():
    """测试检测器初始化（需要配置文件）"""
    # 创建临时配置文件用于测试
    import tempfile
    import yaml
    
    config = {
        'model': {
            'name': 'dinov3_image_level',
            'dino_model_name': 'dinov3_base',
            'model_path': None,
            'num_centers': 50000,
            'buffer_size': 2500000,
            'num_neighbors': 1,
        },
        'data': {
            'train_data_path': './data/train',
            'batch_size': 32,
            'num_workers': 8,
            'image_size': 224,
        },
        'training': {
            'sampling_ratio': None,
        },
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        detector = AnomalyDetector(config_path=config_path)
        assert detector.model_name == "dinov3_image_level"
        assert detector is not None
    finally:
        Path(config_path).unlink()


def test_detector_detect():
    """测试异常检测（需要实际图像和配置文件）"""
    # TODO: 添加实际的测试图像和配置文件
    pass

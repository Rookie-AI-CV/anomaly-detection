"""
Anomaly detector core module.
统一的异常检测接口，支持 cls/patch/combined 三种模式。
"""

import logging
from typing import Union, Optional, Dict, Any, Literal
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class AnomalyDetector:
    
    def __init__(self, config_path: Union[str, Path] = None, checkpoint_path: Optional[Union[str, Path]] = None, config=None):
        from hq_anomaly_detection.core.config import Config
        
        if config is not None:
            self.config = config
        elif config_path:
            self.config = Config.from_file(config_path)
        else:
            raise ValueError("Must provide config_path or config")
        
        self.model = None
        self.model_name = self.config.get("model.name")
        
        # 处理 checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_name' in checkpoint:
                self.config.set("model.dino_model_name", checkpoint['model_name'])
        
        self._init_model()
        
        # 加载 checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model.load_checkpoint(checkpoint_path)
    
    def _init_model(self):
        """初始化模型"""
        if self.model_name != "dinov3_image_level":
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        from hq_anomaly_detection.models.dinov3.image_level import DINOv3ImageLevelDetector
        
        self.model = DINOv3ImageLevelDetector(
            model_name=self.config.get("model.dino_model_name"),
            model_path=self.config.get("model.model_path"),
            num_centers=self.config.get("model.num_centers", 50000),
            buffer_size=self.config.get("model.buffer_size", 2500000),
            num_neighbors=self.config.get("model.num_neighbors", 1),
            reduce_dim=self.config.get("model.reduce_dim"),
            use_faiss=self.config.get("model.use_faiss", True),
            cls_weight=self.config.get("model.cls_weight", 0.5),
        )
        
        self._to_device()
        logger.info(f"Model initialized: {self.config.get('model.dino_model_name')}")
    
    def _to_device(self):
        device_str = self.config.get("training.device", "cuda")
        device_id = self.config.get("training.device_id", 0)
        
        if device_str == "cuda" and torch.cuda.is_available():
            self._device = torch.device(f"cuda:{device_id}")
        else:
            self._device = torch.device("cpu")
        
        self.model = self.model.to(self._device)
        logger.info(f"Model on device: {self._device}")
    
    def _get_device(self) -> torch.device:
        return getattr(self, '_device', torch.device('cpu'))
    
    def _get_mode(self) -> str:
        """获取检测模式：cls / patch / combined"""
        return self.config.get("model.detection_mode", "cls")
    
    # ========== 训练 ==========
    
    def train(self):
        mode = self._get_mode()
        device = self._get_device()
        
        # 创建数据集
        dataloader = self._create_dataloader(is_training=True)
        
        logger.info(f"Training mode: {mode}, device: {device}")
        
        # 使用新的统一训练接口
        self.model.train_on_dataloader(dataloader, mode=mode, device=device)
        
        logger.info(f"Training completed!")
    
    def _create_dataloader(self, is_training: bool = True) -> DataLoader:
        """创建数据加载器"""
        dataset = SimpleImageDataset(
            data_path=self.config.get("data.train_data_path"),
            image_size=self.config.get("data.image_size", 224),
            is_training=is_training,
            use_augmentation=self.config.get("data.use_augmentation", True) and is_training,
        )
        
        device = self._get_device()
        return DataLoader(
            dataset,
            batch_size=self.config.get("data.batch_size", 32),
            shuffle=is_training,
            num_workers=self.config.get("data.num_workers", 4),
            pin_memory=(device.type == "cuda"),
        )
    
    # ========== 推理 ==========
    
    def detect(self, image_path: Union[str, Path], threshold: Optional[float] = None) -> Dict[str, Any]:
        """检测单张图片的异常"""
        mode = self._get_mode()
        cls_weight = self.config.get("model.cls_weight", 0.5)
        
        image_tensor = self._load_image(image_path)
        
        with torch.no_grad():
            result = self.model.predict(
                image_tensor, 
                mode=mode, 
                cls_weight=cls_weight,
                return_anomaly_map=(mode in ['patch', 'combined'])
            )
        
        return self._format_result(result, threshold)
    
    def _load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """加载并预处理图片"""
        image_size = self.config.get("data.image_size", 224)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(self._get_device())
    
    def _format_result(self, result: Dict, threshold: Optional[float]) -> Dict[str, Any]:
        """格式化预测结果"""
        output = {
            "anomaly_score": float(result['anomaly_score'][0]),
            "anomaly_map": result.get('anomaly_map'),
        }
        
        # 添加各级别分数（如果有）
        if 'cls_score' in result:
            output['cls_score'] = float(result['cls_score'][0])
        if 'patch_score' in result:
            output['patch_score'] = float(result['patch_score'][0])
        
        # 阈值判断
        if threshold is not None:
            output['prediction'] = output['anomaly_score'] > threshold
        
        return output
    
    def predict_batch(self, image_paths: list, threshold: Optional[float] = None) -> list:
        """批量预测"""
        return [self.detect(p, threshold) for p in image_paths]
    
    # ========== 保存/加载 ==========
    
    def save_model(self, save_path: Union[str, Path]):
        """保存模型"""
        self.model.save_checkpoint(save_path)
        logger.info(f"Model saved: {save_path}")
    
    def load_model(self, checkpoint_path: Union[str, Path]):
        """加载模型"""
        self.model.load_checkpoint(checkpoint_path)
        self._to_device()
        logger.info(f"Model loaded: {checkpoint_path}")


class SimpleImageDataset(Dataset):
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_size: int = 224,
        is_training: bool = True,
        use_augmentation: bool = True,
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.use_augmentation = use_augmentation and is_training
        
        # 收集图片
        self.image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
            self.image_paths.extend(self.data_path.glob(f'**/*.{ext}'))
            self.image_paths.extend(self.data_path.glob(f'**/*.{ext.upper()}'))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_path}")
        
        logger.info(f"Found {len(self.image_paths)} images")
        
        # 构建 transform
        self.transform = self._build_transform()
    
    def _build_transform(self):
        if self.use_augmentation:
            return transforms.Compose([
                transforms.Resize((int(self.image_size * 1.1), int(self.image_size * 1.1))),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return {
            'image': self.transform(image),
            'image_path': str(image_path),
        }

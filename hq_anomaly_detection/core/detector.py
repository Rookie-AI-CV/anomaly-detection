"""
Anomaly detector core module.

Provides unified interface for anomaly detection models.
Supports DINOv3 image-level anomaly detection.
"""

import logging
from typing import Union, Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Unified anomaly detector interface."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize anomaly detector from config file.
        
        Args:
            config_path: Path to YAML config file
        """
        from hq_anomaly_detection.core.config import Config
        
        self.config_path = Path(config_path)
        self.config = Config.from_file(config_path)
        self.model = None
        
        self.model_name = self.config.get("model.name")
        if self.model_name is None:
            raise ValueError("Config must contain 'model.name' field")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model from config."""
        logger.info(f"Initializing {self.model_name} model...")
        
        if self.model_name == "dinov3_image_level":
            from hq_anomaly_detection.models.dinov3.image_level import DINOv3ImageLevelDetector
            
            dino_model_name = self.config.get("model.dino_model_name")
            model_path = self.config.get("model.model_path") or None
            num_centers = self.config.get("model.num_centers")
            buffer_size = self.config.get("model.buffer_size")
            num_neighbors = self.config.get("model.num_neighbors")
            
            self.model = DINOv3ImageLevelDetector(
                model_name=dino_model_name,
                model_path=model_path,
                num_centers=num_centers,
                buffer_size=buffer_size,
                num_neighbors=num_neighbors,
            )
            
            logger.info(
                f"DINOv3ImageLevelDetector initialized: "
                f"model_name={dino_model_name}, model_path={model_path}"
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def load_model(self, checkpoint_path: Union[str, Path]):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        logger.info(f"Loading model from {checkpoint_path}")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call __init__ first.")
        
        if self.model_name == "dinov3_image_level":
            self.model.load_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Model loading for {self.model_name} not yet implemented.")
    
    def train(self):
        """Train model using batch loading strategy to avoid memory overflow."""
        logger.info(f"Training {self.model_name} model...")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call __init__ first.")
        
        if self.model_name == "dinov3_image_level":
            train_data_path = self.config.get("data.train_data_path")
            batch_size = self.config.get("data.batch_size")
            num_workers = self.config.get("data.num_workers")
            image_size = self.config.get("data.image_size")
            sampling_ratio = self.config.get("training.sampling_ratio") or None
            
            # Get device configuration
            device_str = self.config.get("training.device", "cuda")
            device_id = self.config.get("training.device_id", 0)
            
            if device_str == "cuda" and torch.cuda.is_available():
                device = torch.device(f"cuda:{device_id}")
                logger.info(f"Using GPU: {device}")
                # Move model to GPU
                self.model = self.model.to(device)
            else:
                device = torch.device("cpu")
                logger.info("Using CPU (CUDA not available or device set to cpu)")
                self.model = self.model.to(device)
            
            # Get augmentation configuration
            use_augmentation = self.config.get("data.use_augmentation", True)
            
            # Print training configuration information
            logger.info(f"Training configuration:")
            logger.info(f"  - Device: {device}")
            logger.info(f"  - Image size: {image_size}")
            logger.info(f"  - Batch size: {batch_size}")
            logger.info(f"  - Num workers: {num_workers}")
            logger.info(f"  - Train data path: {train_data_path}")
            logger.info(f"  - Data augmentation: {use_augmentation}")
            
            train_dataset = SimpleImageDataset(
                data_path=train_data_path,
                image_size=image_size,
                is_training=True,
                use_augmentation=use_augmentation
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),  # Use pin_memory only for GPU
            )
            
            logger.info("Extracting features from training data...")
            self.model.extract_features_batch(train_dataloader, device=device)
            
            logger.info("Building memory bank...")
            self.model.build_memory_bank(sampling_ratio)
            
            logger.info("Training completed!")
        else:
            raise ValueError(f"Training for {self.model_name} not yet implemented.")
    
    def detect(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in a single image.
        
        Args:
            image_path: Path to image
            threshold: Optional anomaly threshold
            
        Returns:
            Dict with:
            - anomaly_score: Anomaly score
            - anomaly_map: Anomaly map (None for image-level)
            - prediction: Whether anomaly detected
        """
        logger.debug(f"Detecting anomalies in {image_path}")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call __init__ first.")
        
        if self.model_name == "dinov3_image_level":
            image_size = self.config.get("data.image_size")
            
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image_tensor = transform(image).unsqueeze(0)
            
            device = next(self.model.parameters()).device if next(self.model.parameters()).is_cuda else torch.device('cpu')
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                result = self.model.predict(image_tensor)
            
            if threshold is not None:
                result['prediction'] = result['anomaly_score'] > threshold
            
            return {
                "anomaly_score": float(result['anomaly_score'][0]),
                "anomaly_map": None,
                "prediction": bool(result['prediction'][0]),
            }
        else:
            raise ValueError(f"Detection for {self.model_name} not yet implemented.")
    
    def predict_batch(
        self,
        image_paths: list,
        threshold: Optional[float] = None
    ) -> list:
        """
        Batch prediction.
        
        Args:
            image_paths: List of image paths
            threshold: Optional anomaly threshold
            
        Returns:
            List of detection results
        """
        results = []
        for img_path in image_paths:
            result = self.detect(img_path, threshold)
            results.append(result)
        return results
    
    def save_model(self, save_path: Union[str, Path]):
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save checkpoint
        """
        logger.info(f"Saving model to {save_path}")
        
        if self.model is None:
            raise ValueError("Model not initialized. Call __init__ first.")
        
        if self.model_name == "dinov3_image_level":
            self.model.save_checkpoint(save_path)
        else:
            raise ValueError(f"Model saving for {self.model_name} not yet implemented.")


class SimpleImageDataset(Dataset):
    """Simple image dataset for training and inference."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_size: int = 224,
        is_training: bool = True,
        use_augmentation: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data directory
            image_size: Target image size
            is_training: Whether in training mode
            use_augmentation: Whether to use data augmentation (only for training)
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.is_training = is_training
        self.use_augmentation = use_augmentation and is_training
        
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            self.image_paths.extend(list(self.data_path.glob(f'**/*{ext}')))
            self.image_paths.extend(list(self.data_path.glob(f'**/*{ext.upper()}')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")
        
        logger.info(f"Found {len(self.image_paths)} images in {data_path}")
        
        # Build transform pipeline
        if self.use_augmentation:
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),  # Slightly larger for crop
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),  # Less common in real scenarios
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Optional: random erasing
            ])
            logger.info("Data augmentation enabled for training")
        else:
            # Inference transforms (no augmentation)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            logger.info("Data augmentation disabled (inference mode or augmentation disabled)")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return {
            'image': image_tensor,
            'image_path': str(image_path),
        }

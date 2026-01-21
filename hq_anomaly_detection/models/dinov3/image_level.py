"""
DINOv3-based image-level anomaly detection model.

Uses DINOv3 to extract cls_token as image-level features, then performs anomaly detection.
"""

import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from hq_anomaly_detection.models.dinov3.feature_extractor import DINOv3FeatureExtractor
from hq_anomaly_detection.models.base.memory_bank import KCenterGreedyMemoryBank

logger = logging.getLogger(__name__)


class DINOv3ImageLevelDetector(nn.Module):
    """
    DINOv3-based image-level anomaly detector.
    
    Uses DINOv3 to extract cls_token as image-level features,
    then uses K-NN for anomaly detection.
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_base",
        model_path: Optional[Union[str, Path]] = None,
        num_centers: int = 50000,
        buffer_size: int = 2500000,
        num_neighbors: int = 1,
        **kwargs
    ):
        """
        Initialize DINOv3 image-level anomaly detector.
        
        Args:
            model_name: DINOv3 model name
            model_path: Local model weight path (if provided, don't download automatically)
            num_centers: Maximum number of samples in memory bank
            buffer_size: Maximum size of buffer
            num_neighbors: Number of neighbors for K-NN
            **kwargs: Other parameters
        """
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.num_centers = num_centers
        self.buffer_size = buffer_size
        self.num_neighbors = num_neighbors
        
        # Feature extractor (only uses cls_token)
        self.feature_extractor = DINOv3FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            use_cls_token=True,  # Image-level features
            **kwargs
        )
        
        # Get embedding dimension
        self.embed_dim = self.feature_extractor.embed_dim
        
        # Memory bank
        self.memory_bank: Optional[torch.Tensor] = None
        self.memory_bank_sampler = KCenterGreedyMemoryBank(
            num_centers=num_centers,
            buffer_size=buffer_size,
        )
        
        logger.info(
            f"Initialized DINOv3ImageLevelDetector: "
            f"model={model_name}, embed_dim={self.embed_dim}, "
            f"num_centers={num_centers}, buffer_size={buffer_size}, "
            f"num_neighbors={num_neighbors}"
        )
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract image-level features.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            Image-level features [B, D] (cls_token)
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
        return features
    
    def extract_features_batch(
        self, dataloader: torch.utils.data.DataLoader, device: Optional[torch.device] = None
    ) -> None:
        """
        Extract features in batches and add to memory bank.
        
        Args:
            dataloader: Data loader
            device: Device (if None, auto-detect)
        """
        self.feature_extractor.eval()
        
        # Auto-detect device (if not provided)
        if device is None:
            device = next(self.feature_extractor.parameters()).device
        
        logger.info(f"Starting feature extraction on {device}...")
        total_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Get images
                if isinstance(batch, dict):
                    images = batch['image']
                elif isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                # Move images to device
                images = images.to(device)
                
                # Extract features
                features = self.feature_extractor(images)
                
                # Add to memory bank (move to CPU)
                self.memory_bank_sampler.add_embeddings(features.cpu())
                
                total_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    stats = self.memory_bank_sampler.get_statistics()
                    logger.info(
                        f"Processed {batch_idx + 1} batches, "
                        f"total embeddings: {stats['total_embeddings']}, "
                        f"buffer usage: {stats['buffer_usage']:.2%}"
                    )
        
        logger.info(f"Feature extraction completed. Total batches: {total_batches}")
    
    def build_memory_bank(
        self, sampling_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        Build memory bank.
        
        Args:
            sampling_ratio: Sampling ratio (if None, use num_centers)
            
        Returns:
            Sampled memory bank [N, D]
        """
        sampled_embeddings = self.memory_bank_sampler.sample(sampling_ratio)
        self.memory_bank = sampled_embeddings
        logger.info(
            f"Memory bank built with {sampled_embeddings.shape[0]} samples, "
            f"embedding dimension: {sampled_embeddings.shape[1]}"
        )
        return sampled_embeddings
    
    def predict(
        self, input_tensor: torch.Tensor, return_anomaly_map: bool = False
    ) -> Dict[str, Any]:
        """
        Predict anomalies.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            return_anomaly_map: Whether to return anomaly map (False for image-level)
            
        Returns:
            Prediction result dictionary:
            - anomaly_score: Anomaly score [B]
            - anomaly_map: Anomaly map (None for image-level)
            - prediction: Prediction result [B] (bool)
        """
        if self.memory_bank is None:
            raise ValueError(
                "Memory bank not built. Call build_memory_bank() first."
            )
        
        # Extract features
        features = self(input_tensor)  # [B, D]
        
        # Calculate distance to memory bank
        device = features.device
        memory_bank = self.memory_bank.to(device)
        
        # Calculate distances [B, N]
        distances = torch.cdist(features, memory_bank)
        
        # Find K nearest neighbors
        k = min(self.num_neighbors, memory_bank.shape[0])
        kth_distances, _ = torch.topk(distances, k, dim=1, largest=False)
        
        # Anomaly score = mean distance to K nearest neighbors
        anomaly_score = kth_distances.mean(dim=1)  # [B]
        
        # Prediction (can be adjusted based on threshold)
        prediction = anomaly_score > 0.5  # Simple threshold, should be determined based on validation set
        
        result = {
            "anomaly_score": anomaly_score.cpu().numpy(),
            "anomaly_map": None,  # No spatial map for image-level
            "prediction": prediction.cpu().numpy(),
        }
        
        return result
    
    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        """
        Set memory bank.
        
        Args:
            memory_bank: Memory bank tensor [N, D]
        """
        self.memory_bank = memory_bank
        logger.info(
            f"Memory bank set with {memory_bank.shape[0]} samples, "
            f"embedding dimension: {memory_bank.shape[1]}"
        )
    
    def save_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Save checkpoint.
        
        Args:
            checkpoint_path: Checkpoint save path
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_name': self.model_name,
            'model_path': str(self.model_path) if self.model_path else None,
            'num_centers': self.num_centers,
            'buffer_size': self.buffer_size,
            'num_neighbors': self.num_neighbors,
            'embed_dim': self.embed_dim,
            'memory_bank': self.memory_bank,
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Checkpoint path
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load memory bank
        if 'memory_bank' in checkpoint:
            self.memory_bank = checkpoint['memory_bank']
        
        # Load feature extractor weights (if needed)
        if 'feature_extractor_state_dict' in checkpoint:
            self.feature_extractor.load_state_dict(
                checkpoint['feature_extractor_state_dict'], strict=False
            )
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

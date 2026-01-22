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
        
        # 特征记忆库（cls_token），初始化为None
        self.memory_bank: Optional[torch.Tensor] = None
        # 特征记忆库采样器，使用K-Center Greedy算法，初始化为None
        self.memory_bank_sampler = KCenterGreedyMemoryBank(
            num_centers=num_centers,
            buffer_size=buffer_size,
        )
        
        # Patch 级特征提取器（用于PatchCore风格的检测），初始化为None
        self.patch_feature_extractor: Optional[DINOv3FeatureExtractor] = None
        # Patch 级记忆库（特征张量），初始化为None
        self.patch_memory_bank: Optional[torch.Tensor] = None
        # Patch 级记忆库采样器，使用K-Center Greedy算法，初始化为None
        self.patch_memory_bank_sampler: Optional[KCenterGreedyMemoryBank] = None   
        logger.info(
            f"Initialized DINOv3ImageLevelDetector: "
            f"model={model_name}, embed_dim={self.embed_dim}, "
            f"num_centers={num_centers}, buffer_size={buffer_size}, "
            f"num_neighbors={num_neighbors}"
        )
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract image-level features. 提取图像级别特征
        
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
        分批提取特征并将其添加到特征记忆库中
        该方法遍历 dataloader，将每批图像输入特征提取器，获得特征后存入 memory_bank_sampler
        适用于大规模数据时的内存高效特征收集，便于后续采样/构建 KNN 记忆库
        
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
        构建特征记忆库
        使用 K-Center Greedy 算法从特征记忆库采样器中采样，构建特征记忆库。
        
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
        Predict anomalies. 预测异常
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            return_anomaly_map: 是否返回异常图 (False for image-level)
        """
        if self.memory_bank is None:
            raise ValueError(
                "Memory bank not built. Call build_memory_bank() first."
            )
        
        # 提取特征
        features = self(input_tensor)  # [B, D]
        
        # 计算特征与记忆库中所有特征的距离
        device = features.device
        memory_bank = self.memory_bank.to(device)
        
        # 计算特征与记忆库中所有特征的距离 [B, N]
        distances = torch.cdist(features, memory_bank)
        
        # 找到 K 个最近邻特征
        k = min(self.num_neighbors, memory_bank.shape[0])
        kth_distances, _ = torch.topk(distances, k, dim=1, largest=False)
        
        # 异常分数 = K 个最近邻距离的平均值
        anomaly_score = kth_distances.mean(dim=1)  # [B]
        
        result = {
            "anomaly_score": anomaly_score.cpu().numpy(),
            "anomaly_map": None,  # 图像级别检测无空间图
        }
        
        return result
    
    def set_memory_bank(self, memory_bank: torch.Tensor) -> None:
        """
        Set特征记忆库
        
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
            'patch_memory_bank': self.patch_memory_bank,
        }
        
        # Save patch feature extractor state if it exists
        if self.patch_feature_extractor is not None:
            checkpoint['patch_feature_extractor_state_dict'] = self.patch_feature_extractor.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load 加载检查点
        
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
        
        # Load patch memory bank
        if 'patch_memory_bank' in checkpoint:
            self.patch_memory_bank = checkpoint['patch_memory_bank']
        
        # Load feature extractor weights (if needed)
        if 'feature_extractor_state_dict' in checkpoint:
            self.feature_extractor.load_state_dict(
                checkpoint['feature_extractor_state_dict'], strict=False
            )
        
        # Load patch feature extractor weights (if needed)
        if 'patch_feature_extractor_state_dict' in checkpoint:
            patch_extractor = self._get_patch_feature_extractor()
            patch_extractor.load_state_dict(
                checkpoint['patch_feature_extractor_state_dict'], strict=False
            )
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _get_patch_feature_extractor(self) -> DINOv3FeatureExtractor:
        """
        获取或创建 patch 级特征提取器
        
        Returns:
            Patch-level feature extractor (uses patch tokens, not cls_token)
        """
        if self.patch_feature_extractor is None:
            self.patch_feature_extractor = DINOv3FeatureExtractor(
                model_name=self.model_name,
                model_path=self.model_path,
                use_cls_token=False,  # Use patch tokens
            )
        return self.patch_feature_extractor
    
    def _get_patch_memory_bank_sampler(self) -> KCenterGreedyMemoryBank:
        """
        Get or create patch-level memory bank sampler.
        
        Returns:
            Patch-level memory bank sampler
        """
        if self.patch_memory_bank_sampler is None:
            self.patch_memory_bank_sampler = KCenterGreedyMemoryBank(
                num_centers=self.num_centers,
                buffer_size=self.buffer_size,
            )
        return self.patch_memory_bank_sampler
    
    def extract_patch_features_batch(
        self, dataloader: torch.utils.data.DataLoader, device: Optional[torch.device] = None
    ) -> None:
        """
        Extract patch-level features in batches and add to patch memory bank.
        Used for PatchCore-style detection.
        
        Args:
            dataloader: Data loader
            device: Device (if None, auto-detect)
        """
        patch_extractor = self._get_patch_feature_extractor()
        patch_extractor.eval()
        
        # Auto-detect device (if not provided)
        if device is None:
            device = next(self.feature_extractor.parameters()).device
        
        # Ensure patch_extractor is on the correct device
        patch_extractor = patch_extractor.to(device)
        
        patch_sampler = self._get_patch_memory_bank_sampler()
        
        logger.info(f"Starting patch-level feature extraction on {device}...")
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
                
                # Extract patch-level features [B*H*W, D]
                try:
                    patch_features = patch_extractor(images)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"GPU out of memory at batch {batch_idx}. Try reducing batch_size or image_size.")
                        logger.error(f"Current batch_size: {images.shape[0]}, image_size: {images.shape[-1]}")
                        raise
                    else:
                        raise
                
                # Add to patch memory bank (move to CPU)
                patch_sampler.add_embeddings(patch_features.cpu())
                
                total_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    stats = patch_sampler.get_statistics()
                    logger.info(
                        f"Processed {batch_idx + 1} batches, "
                        f"total patch embeddings: {stats['total_embeddings']}, "
                        f"buffer usage: {stats['buffer_usage']:.2%}"
                    )
        
        logger.info(f"Patch-level feature extraction completed. Total batches: {total_batches}")
    
    def build_patch_memory_bank(
        self, sampling_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        构建 PatchCore 风格检测的 patch 级记忆库
        使用 K-Center Greedy 算法从 patch 级记忆库采样器中采样，构建 patch 级记忆库。
        Args:
            sampling_ratio: Sampling ratio (if None, use num_centers)
            
        Returns:
            Sampled patch memory bank [N, D]
        """
        patch_sampler = self._get_patch_memory_bank_sampler()
        sampled_embeddings = patch_sampler.sample(sampling_ratio)
        self.patch_memory_bank = sampled_embeddings
        logger.info(
            f"Patch memory bank built with {sampled_embeddings.shape[0]} patch samples, "
            f"embedding dimension: {sampled_embeddings.shape[1]}"
        )
        return sampled_embeddings
    
    def predict_patchcore_style(
        self, input_tensor: torch.Tensor, return_anomaly_map: bool = False
    ) -> Dict[str, Any]:
        """
        使用 PatchCore 风格方法预测异常
        使用 patch 级特征和计算：
        1. 对于每个 patch：距离最近的邻居在记忆库中的距离（1-NN）
        2. 图像级别分数：所有 patch 分数的最大值（max pooling）
        
        遵循 PatchCore 逻辑，但使用 DINOv3 patch tokens 而不是 CNN 特征
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            return_anomaly_map: Whether to return anomaly map (can be used for visualization)
            
        Returns:
            Prediction result dictionary:
            - anomaly_score: Anomaly score [B] (max of patch scores)
            - anomaly_map: Anomaly map [B, H, W] (patch-level scores reshaped) or None
            - prediction: Prediction result [B] (bool)
        """
        if self.patch_memory_bank is None:
            raise ValueError(
                "Patch memory bank not built. Call build_patch_memory_bank() first."
            )
        
        patch_extractor = self._get_patch_feature_extractor()
        patch_extractor.eval()
        
        device = input_tensor.device
        patch_extractor = patch_extractor.to(device)
        
        # 提取 patch 级特征 [B*H*W, D]
        with torch.no_grad():
            patch_features = patch_extractor(input_tensor)
        patch_memory_bank = self.patch_memory_bank.to(device)
        
        # 获取批量大小和 patch 维度
        batch_size = input_tensor.shape[0]
        num_patches_per_image = patch_features.shape[0] // batch_size
        
        # 计算每个 patch 与记忆库中所有特征的距离 [B*H*W, N]
        distances = torch.cdist(patch_features, patch_memory_bank)
        
        # 找到每个 patch 的最近邻特征 (1-NN, 类似于 PatchCore)
        # distances shape: [B*H*W, N]
        min_distances, _ = torch.min(distances, dim=1)  # [B*H*W]
        
        # 重塑为 [B, H*W] 以获取每个图像的 patch 分数
        patch_scores = min_distances.reshape(batch_size, num_patches_per_image)
        
        # 图像级别异常分数 = patch 分数的最大值 (max pooling, 类似于 PatchCore)
        anomaly_score = torch.max(patch_scores, dim=1)[0]  # [B]
        
        result = {
            "anomaly_score": anomaly_score.cpu().numpy(),
        }
        
        # 可选返回异常图 (patch 级分数重塑为空间维度)
        if return_anomaly_map:
            # 尝试从 patch 数量推断空间维度
            # 对于 DINOv3，patches 通常排列在网格中
            # num_patches = H_patch * W_patch, where H_patch = H / patch_size, W_patch = W / patch_size
            # 我们将重塑为近似空间布局，注意：这可能是近似的，可能不匹配精确的空间位置
            patch_h = int(num_patches_per_image ** 0.5)  # 近似方形网格
            patch_w = num_patches_per_image // patch_h
            
            if patch_h * patch_w == num_patches_per_image:
                anomaly_map = patch_scores.reshape(batch_size, patch_h, patch_w)
                result["anomaly_map"] = anomaly_map.cpu().numpy()
            else:
                # If not perfect square, return flattened version
                logger.warning(
                    f"Cannot reshape {num_patches_per_image} patches to square grid. "
                    "Returning flattened anomaly map."
                )
                result["anomaly_map"] = patch_scores.cpu().numpy()
        else:
            result["anomaly_map"] = None
        
        return result
    
    def set_patch_memory_bank(self, patch_memory_bank: torch.Tensor) -> None:
        """
        Set patch-level memory bank. 设置 patch 级记忆库
        
        Args:
            patch_memory_bank: Patch memory bank tensor [N, D]
        """
        self.patch_memory_bank = patch_memory_bank
        logger.info(
            f"Patch memory bank set with {patch_memory_bank.shape[0]} patch samples, "
            f"embedding dimension: {patch_memory_bank.shape[1]}"
        )

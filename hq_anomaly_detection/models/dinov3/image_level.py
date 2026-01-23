import logging
from typing import Optional, Dict, Any, Union, Literal
from pathlib import Path
import torch
from torch import nn
import numpy as np

from hq_anomaly_detection.models.dinov3.feature_extractor import DINOv3FeatureExtractor
from hq_anomaly_detection.models.base.memory_bank import KCenterGreedyMemoryBank, RandomProjection

logger = logging.getLogger(__name__)


# ============================================================================
# FeatureLevel: 单个特征级别的封装（cls_token 或 patch_token）
# ============================================================================

class FeatureLevel:
    """
    单个特征级别的封装，统一 cls_token 和 patch_token 的处理逻辑
    
    包含：特征提取器、记忆库、采样器、投影器
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        model_path: Optional[str],
        use_cls_token: bool,
        num_centers: int,
        buffer_size: int,
        reduce_dim: Optional[int],
        use_faiss: bool,
        device: Optional[Union[str, torch.device]] = None,
        use_gpu: bool = True,
    ):
        self.name = name
        self.use_cls_token = use_cls_token
        
        # 设备管理
        if device is None:
            self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 特征提取器
        self.feature_extractor = DINOv3FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            use_cls_token=use_cls_token,
        )
        self.embed_dim = self.feature_extractor.embed_dim
        
        # 记忆库相关（传递设备信息）
        self.memory_bank: Optional[torch.Tensor] = None
        self.memory_bank_sampler = KCenterGreedyMemoryBank(
            num_centers=num_centers,
            buffer_size=buffer_size,
            reduce_dim=reduce_dim,
            use_faiss=use_faiss,
            device=self.device,
            use_gpu=use_gpu,
        )
        self.projector: Optional[RandomProjection] = None
        self.image_names: Optional[list] = None
    
    def to(self, device):
        self.device = torch.device(device)
        self.feature_extractor = self.feature_extractor.to(device)
        # 更新 memory_bank_sampler 的设备（如果支持）
        if hasattr(self.memory_bank_sampler, 'device'):
            self.memory_bank_sampler.device = self.device
        return self
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        self.feature_extractor.eval()
        with torch.no_grad():
            return self.feature_extractor(images)
    
    def add_to_memory_bank(self, features: torch.Tensor, image_names: Optional[list] = None):
        """添加特征到采样器"""
        self.memory_bank_sampler.add_embeddings(features, image_names)
    
    def build_memory_bank(self, sampling_ratio: Optional[float] = None) -> torch.Tensor:
        """构建记忆库"""
        self.memory_bank = self.memory_bank_sampler.sample(sampling_ratio)
        self.projector = self.memory_bank_sampler.get_projector()
        self.image_names = self.memory_bank_sampler.get_image_names()
        logger.info(f"[{self.name}] Memory bank: {self.memory_bank.shape[0]} samples, dim={self.memory_bank.shape[1]}")
        return self.memory_bank
    
    def compute_anomaly_score(
        self, 
        features: torch.Tensor, 
        num_neighbors: int = 1,
        aggregation: str = 'mean'  # 'mean' for cls_token, 'max' for patch_token
    ) -> torch.Tensor:
        """
        计算异常分数
        
        Args:
            features: 特征 [N, D] 或 [B*P, D]
            num_neighbors: K-NN 的 K
            aggregation: 聚合方式 ('mean' 或 'max')
        
        Returns:
            异常分数 [N] 或 [B]
        """
        if self.memory_bank is None:
            raise ValueError(f"[{self.name}] Memory bank not built")
        
        # 应用投影（如果有）
        if self.projector is not None:
            features = self.projector.transform(features.cpu()).to(features.device)
        
        memory_bank = self.memory_bank.to(features.device)
        distances = torch.cdist(features, memory_bank)
        
        if aggregation == 'mean':
            # cls_token: K-NN 距离的均值
            k = min(num_neighbors, memory_bank.shape[0])
            kth_distances, _ = torch.topk(distances, k, dim=1, largest=False)
            return kth_distances.mean(dim=1)
        else:
            # patch_token: 1-NN 距离
            min_distances, _ = torch.min(distances, dim=1)
            return min_distances
    
    def get_state(self) -> dict:
        """获取状态用于保存"""
        return {
            'memory_bank': self.memory_bank,
            'projector': self.projector,
            'image_names': self.image_names,
        }
    
    def load_state(self, state: dict):
        """加载状态"""
        self.memory_bank = state.get('memory_bank')
        self.projector = state.get('projector')
        self.image_names = state.get('image_names')


# ============================================================================
# DINOv3AnomalyDetector: 主检测器，支持 cls/patch/combined 三种模式
# ============================================================================

class DINOv3AnomalyDetector(nn.Module):
    """
    DINOv3 异常检测器
    
    支持三种检测模式：
    - 'cls': 仅使用 cls_token（图像级别）
    - 'patch': 仅使用 patch_token（PatchCore 风格）
    - 'combined': 联合判定，加权融合两个异常分数
    
    Usage:
        detector = DINOv3AnomalyDetector(model_name='dinov3_base')
        
        # 训练
        detector.train_on_dataloader(dataloader, mode='combined')
        
        # 推理
        result = detector.predict(images, mode='combined', cls_weight=0.5)
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_base",
        model_path: Optional[Union[str, Path]] = None,
        num_centers: int = 50000,
        buffer_size: int = 2500000,
        num_neighbors: int = 1,
        reduce_dim: Optional[int] = None,
        use_faiss: bool = True,
        cls_weight: float = 0.5,  # 联合判定时 cls_token 的权重
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.model_path = str(model_path) if model_path else None
        self.num_neighbors = num_neighbors
        self.cls_weight = cls_weight
        
        # CLS 级别
        self.cls_level = FeatureLevel(
            name='cls',
            model_name=model_name,
            model_path=self.model_path,
            use_cls_token=True,
            num_centers=num_centers,
            buffer_size=buffer_size,
            reduce_dim=reduce_dim,
            use_faiss=use_faiss,
        )
        
        # Patch 级别
        self.patch_level = FeatureLevel(
            name='patch',
            model_name=model_name,
            model_path=self.model_path,
            use_cls_token=False,
            num_centers=num_centers,
            buffer_size=buffer_size,
            reduce_dim=reduce_dim,
            use_faiss=use_faiss,
        )
        
        self.image_dir_path: Optional[str] = None
        
        logger.info(f"DINOv3AnomalyDetector: model={model_name}, cls_weight={cls_weight}")
    
    def to(self, device):
        """移动模型到指定设备"""
        self.cls_level.to(device)
        self.patch_level.to(device)
        return super().to(device)
    
    # ========== 训练相关 ==========
    
    def train_on_dataloader(
        self, 
        dataloader, 
        mode: Literal['cls', 'patch', 'combined'] = 'combined',
        device: Optional[torch.device] = None,
    ) -> None:
        """
        在 dataloader 上训练（提取特征并构建记忆库）
        
        Args:
            dataloader: 数据加载器
            mode: 'cls' / 'patch' / 'combined'
            device: 计算设备
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if mode in ['cls', 'combined']:
            logger.info("Extracting CLS features...")
            self._extract_features(self.cls_level, dataloader, device)
            self.cls_level.build_memory_bank()
        
        if mode in ['patch', 'combined']:
            logger.info("Extracting Patch features...")
            self._extract_features(self.patch_level, dataloader, device)
            self.patch_level.build_memory_bank()
    
    def _extract_features(self, level: FeatureLevel, dataloader, device) -> None:
        """提取特征到指定级别的记忆库"""
        level.to(device)
        all_image_paths = []
        
        for batch_idx, batch in enumerate(dataloader):
            images, image_paths = self._parse_batch(batch)
            if image_paths:
                all_image_paths.extend(image_paths if isinstance(image_paths, list) else [image_paths])
            
            features = level.extract_features(images.to(device))
            
            # 对于 patch 级别，不需要 image_names
            if level.use_cls_token:
                names = list(image_paths) if image_paths else [f"img_{batch_idx}_{i}" for i in range(len(features))]
                level.add_to_memory_bank(features, names)
            else:
                level.add_to_memory_bank(features)
            
            if (batch_idx + 1) % 50 == 0:
                stats = level.memory_bank_sampler.get_statistics()
                logger.info(f"[{level.name}] Batch {batch_idx + 1}: {stats['total_embeddings']} embeddings")
        
        if all_image_paths and level.use_cls_token:
            self.image_dir_path = self._find_common_path(all_image_paths)
    
    # ========== 推理相关 ==========
    
    def predict(
        self, 
        input_tensor: torch.Tensor,
        mode: Literal['cls', 'patch', 'combined'] = 'combined',
        cls_weight: Optional[float] = None,
        return_anomaly_map: bool = False,
    ) -> Dict[str, Any]:
        """
        预测异常分数
        
        Args:
            input_tensor: 输入图像 [B, C, H, W]
            mode: 'cls' / 'patch' / 'combined'
            cls_weight: 联合判定时 cls 的权重（None 则使用初始化时的值）
            return_anomaly_map: 是否返回异常热图（仅 patch 模式）
        
        Returns:
            dict: {
                'anomaly_score': np.ndarray [B],
                'cls_score': np.ndarray [B] (if mode != 'patch'),
                'patch_score': np.ndarray [B] (if mode != 'cls'),
                'anomaly_map': np.ndarray [B, H, W] (if return_anomaly_map),
            }
        """
        cls_weight = cls_weight if cls_weight is not None else self.cls_weight
        device = input_tensor.device
        batch_size = input_tensor.shape[0]
        result = {}
        
        # CLS 级别
        if mode in ['cls', 'combined']:
            self.cls_level.to(device)
            cls_features = self.cls_level.extract_features(input_tensor)
            cls_score = self.cls_level.compute_anomaly_score(
                cls_features, self.num_neighbors, aggregation='mean'
            )
            result['cls_score'] = cls_score.cpu().numpy()
        
        # Patch 级别
        if mode in ['patch', 'combined']:
            self.patch_level.to(device)
            patch_features = self.patch_level.extract_features(input_tensor)
            num_patches = patch_features.shape[0] // batch_size
            
            patch_distances = self.patch_level.compute_anomaly_score(
                patch_features, aggregation='max'  # 返回每个 patch 的 1-NN 距离
            )
            patch_scores = patch_distances.reshape(batch_size, num_patches)
            patch_score = torch.max(patch_scores, dim=1)[0]  # max pooling
            result['patch_score'] = patch_score.cpu().numpy()
            
            if return_anomaly_map:
                patch_h = int(num_patches ** 0.5)
                patch_w = num_patches // patch_h
                if patch_h * patch_w == num_patches:
                    result['anomaly_map'] = patch_scores.reshape(batch_size, patch_h, patch_w).cpu().numpy()
                else:
                    result['anomaly_map'] = patch_scores.cpu().numpy()
        
        # 计算最终异常分数
        if mode == 'cls':
            result['anomaly_score'] = result['cls_score']
        elif mode == 'patch':
            result['anomaly_score'] = result['patch_score']
        else:  # combined
            result['anomaly_score'] = (
                cls_weight * result['cls_score'] + 
                (1 - cls_weight) * result['patch_score']
            )
        
        return result
    
    # ========== 保存/加载 ==========
    
    def save_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """保存检查点"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_name': self.model_name,
            'num_neighbors': self.num_neighbors,
            'cls_weight': self.cls_weight,
            'image_dir_path': self.image_dir_path,
            'cls_level': self.cls_level.get_state(),
            'patch_level': self.patch_level.get_state(),
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.cls_weight = checkpoint.get('cls_weight', self.cls_weight)
        self.image_dir_path = checkpoint.get('image_dir_path')
        
        if 'cls_level' in checkpoint:
            self.cls_level.load_state(checkpoint['cls_level'])
        if 'patch_level' in checkpoint:
            self.patch_level.load_state(checkpoint['patch_level'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    # ========== 工具方法 ==========
    
    def _parse_batch(self, batch):
        """解析 batch 数据"""
        if isinstance(batch, dict):
            return batch['image'], batch.get('image_path', [])
        elif isinstance(batch, (list, tuple)):
            return batch[0], batch[1] if len(batch) > 1 else []
        return batch, []
    
    def _find_common_path(self, paths):
        """找到路径列表的公共父目录"""
        if not paths:
            return None
        path_objs = [Path(p).parent for p in paths]
        common = path_objs[0]
        for p in path_objs[1:]:
            common_parts = [a for a, b in zip(common.parts, p.parts) if a == b]
            common = Path(*common_parts) if common_parts else common
        return str(common)


class DINOv3ImageLevelDetector(DINOv3AnomalyDetector):
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.cls_level.extract_features(input_tensor)
    
    # 旧 API 兼容
    @property
    def memory_bank(self):
        return self.cls_level.memory_bank
    
    @property
    def patch_memory_bank(self):
        return self.patch_level.memory_bank
    
    @property
    def embed_dim(self):
        return self.cls_level.embed_dim
    
    def extract_features_batch(self, dataloader, device=None):
        self._extract_features(self.cls_level, dataloader, device or torch.device('cuda'))
    
    def build_memory_bank(self, sampling_ratio=None):
        return self.cls_level.build_memory_bank(sampling_ratio)
    
    def extract_patch_features_batch(self, dataloader, device=None):
        self._extract_features(self.patch_level, dataloader, device or torch.device('cuda'))
    
    def build_patch_memory_bank(self, sampling_ratio=None):
        return self.patch_level.build_memory_bank(sampling_ratio)
    
    def predict_patchcore_style(self, input_tensor, return_anomaly_map=False):
        return self.predict(input_tensor, mode='patch', return_anomaly_map=return_anomaly_map)

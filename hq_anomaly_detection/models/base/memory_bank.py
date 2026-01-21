"""
内存银行模块

实现分批加载数据策略，优化内存使用。
参考 projects/PatchCore 的实现。
"""

import logging
from typing import Optional
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class KCenterGreedyMemoryBank:
    """
    K-Center Greedy 内存银行
    
    实现分批加载数据策略，避免内存溢出。
    当 buffer 满时会自动采样，然后再添加新的 embedding。
    """
    
    def __init__(
        self,
        num_centers: int = 50000,
        buffer_size: int = 2500000,
    ):
        """
        初始化内存银行
        
        Args:
            num_centers: 内存银行中的最大样本数
            buffer_size: 缓冲区的最大大小（用于累积 embedding）
        """
        self.num_centers = num_centers
        self.buffer_size = buffer_size
        self.embeddings: Optional[Tensor] = None
        
        logger.info(
            f"Initialized KCenterGreedyMemoryBank: "
            f"num_centers={num_centers}, buffer_size={buffer_size}"
        )
    
    def add_embeddings(self, embeddings: Tensor) -> None:
        """
        添加新的 embedding 到缓冲区
        
        当缓冲区满时，会自动采样后再添加新的 embedding。
        
        Args:
            embeddings: 新的 embedding [N, D]
        """
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            if self.embeddings.shape[0] < self.buffer_size:
                # 缓冲区未满，直接添加
                self.embeddings = torch.vstack([self.embeddings, embeddings])
            else:
                # 缓冲区已满，需要采样后再添加
                logger.warning(
                    f"Buffer full ({self.embeddings.shape[0]} >= {self.buffer_size}). "
                    "Sampling before adding new embeddings."
                )
                self.sample()
                # 添加新的 embedding
                if self.embeddings.shape[0] + embeddings.shape[0] <= self.buffer_size:
                    self.embeddings = torch.vstack([self.embeddings, embeddings])
                else:
                    # 仍然太大，需要合并后重新采样
                    temp_embeddings = torch.vstack([self.embeddings, embeddings])
                    sampling_ratio = self.num_centers / temp_embeddings.shape[0]
                    self.embeddings = self._k_center_greedy_sample(
                        temp_embeddings, sampling_ratio
                    )
        
        logger.debug(
            f"Added {embeddings.shape[0]} embeddings. "
            f"Total: {self.embeddings.shape[0] if self.embeddings is not None else 0}"
        )
    
    def sample(self, sampling_ratio: Optional[float] = None) -> Tensor:
        """
        从累积的 embedding 中采样构建内存银行
        
        Args:
            sampling_ratio: 采样比例（如果为 None，使用 num_centers）
            
        Returns:
            采样后的 embedding [num_centers, D]
        """
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            raise ValueError("No embeddings to sample. Add embeddings first.")
        
        if sampling_ratio is None:
            sampling_ratio = self.num_centers / self.embeddings.shape[0]
        
        # 限制采样比例在 [0, 1] 之间
        sampling_ratio = max(0.0, min(1.0, sampling_ratio))
        
        # 如果 embedding 数量已经少于 num_centers，返回所有
        if self.embeddings.shape[0] <= self.num_centers:
            logger.info(
                f"Embeddings ({self.embeddings.shape[0]}) <= num_centers ({self.num_centers}). "
                "Returning all embeddings."
            )
            return self.embeddings
        
        # 使用 K-Center Greedy 采样
        sampled_embeddings = self._k_center_greedy_sample(
            self.embeddings, sampling_ratio
        )
        
        # 更新内部 embedding
        self.embeddings = sampled_embeddings
        
        logger.info(
            f"Sampled {sampled_embeddings.shape[0]} embeddings from "
            f"{self.embeddings.shape[0]} total embeddings."
        )
        
        return sampled_embeddings
    
    def _k_center_greedy_sample(
        self, embeddings: Tensor, sampling_ratio: float
    ) -> Tensor:
        """
        使用 K-Center Greedy 算法采样
        
        Args:
            embeddings: 输入 embedding [N, D]
            sampling_ratio: 采样比例
            
        Returns:
            采样后的 embedding [M, D]，其中 M = int(N * sampling_ratio)
        """
        num_samples = int(embeddings.shape[0] * sampling_ratio)
        num_samples = max(1, min(num_samples, embeddings.shape[0]))
        
        if num_samples >= embeddings.shape[0]:
            return embeddings
        
        # 使用简单的 K-Center Greedy 实现
        # 选择第一个样本（随机或第一个）
        selected_indices = [0]
        distances = torch.cdist(embeddings, embeddings[0:1])[:, 0]
        
        # 迭代选择剩余的样本
        for _ in range(1, num_samples):
            # 找到距离已选择样本集合最远的点
            min_distances = torch.min(
                torch.cdist(embeddings, embeddings[selected_indices]), dim=1
            )[0]
            # 选择距离最远的点
            farthest_idx = torch.argmax(min_distances).item()
            selected_indices.append(farthest_idx)
        
        return embeddings[selected_indices]
    
    def reset(self) -> None:
        """重置内存银行缓冲区"""
        self.embeddings = None
        logger.info("Memory bank reset.")
    
    def get_embeddings(self) -> Optional[Tensor]:
        """
        获取当前的 embedding 缓冲区
        
        Returns:
            当前的 embedding 或 None
        """
        return self.embeddings
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_embeddings': (
                self.embeddings.shape[0] if self.embeddings is not None else 0
            ),
            'embedding_dim': (
                self.embeddings.shape[1] if self.embeddings is not None else 0
            ),
            'buffer_size': self.buffer_size,
            'num_centers': self.num_centers,
            'buffer_usage': (
                self.embeddings.shape[0] / self.buffer_size
                if self.embeddings is not None and self.buffer_size > 0
                else 0.0
            ),
        }
        return stats

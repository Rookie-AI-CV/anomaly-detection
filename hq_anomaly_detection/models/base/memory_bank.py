import logging
from typing import Optional, List, Union
import torch
from torch import Tensor
import numpy as np
import faiss
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm

logger = logging.getLogger(__name__)

FAISS_GPU_AVAILABLE = False
if torch.cuda.is_available():
    try:
        if hasattr(faiss, 'GpuIndexFlatL2') and hasattr(faiss, 'StandardGpuResources'):
            FAISS_GPU_AVAILABLE = True
    except (AttributeError, ImportError):
        FAISS_GPU_AVAILABLE = False


class RandomProjection:
    """随机投影降维"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        self.projector = SparseRandomProjection(n_components=output_dim, random_state=42)
        self.fitted = False
    
    def fit(self, X: Tensor) -> 'RandomProjection':
        X_np = X.cpu().numpy() if isinstance(X, Tensor) else X
        self.projector.fit(X_np)
        self.fitted = True
        return self
    
    def transform(self, X: Tensor) -> Tensor:
        X_np = X.cpu().numpy() if isinstance(X, Tensor) else X
        X_reduced = self.projector.transform(X_np)
        return torch.from_numpy(X_reduced).float()
    
    def fit_transform(self, X: Tensor) -> Tensor:
        self.fit(X)
        return self.transform(X)


class KCenterGreedyMemoryBank:
    """K-Center Greedy Memory Bank with optional dimension reduction and GPU acceleration"""
    
    def __init__(
        self,
        num_centers: int = 50000,
        buffer_size: int = 2500000,
        reduce_dim: Optional[int] = None,
        use_faiss: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        use_gpu: bool = True,
        adaptive_batch_size: bool = True,
        max_batch_size: int = 1024,
    ):
        self.num_centers = num_centers
        self.buffer_size = buffer_size
        self.reduce_dim = reduce_dim
        self.use_faiss = use_faiss
        self.adaptive_batch_size = adaptive_batch_size
        self.max_batch_size = max_batch_size
        
        # 设备管理
        if device is None:
            self.device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.use_gpu = use_gpu and torch.cuda.is_available() and self.device.type == 'cuda'
        self.use_faiss_gpu = self.use_faiss and self.use_gpu and FAISS_GPU_AVAILABLE
        
        self.embeddings: Optional[Tensor] = None
        self.image_names: List[str] = []
        self.projector: Optional[RandomProjection] = None
        self.original_dim: Optional[int] = None
        
        # 自适应批量大小
        self._optimal_batch_size: Optional[int] = None
        
        logger.info(
            f"KCenterGreedyMemoryBank: num_centers={num_centers}, reduce_dim={reduce_dim}, "
            f"device={self.device}, use_gpu={self.use_gpu}, faiss_gpu={self.use_faiss_gpu}, "
            f"adaptive_batch={adaptive_batch_size}, max_batch={max_batch_size}"
        )
    
    def _get_gpu_memory_info(self) -> tuple:
        """获取GPU显存信息 (已用, 总计, 可用)"""
        if not self.use_gpu:
            return (0, 0, 0)
        try:
            total = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            free = total - reserved
            return (allocated, total, free)
        except:
            return (0, 0, 0)
    
    def _estimate_optimal_batch_size(self, n: int, d: int) -> int:
        """根据GPU显存自适应估算最优批量大小"""
        if not self.use_gpu or not self.adaptive_batch_size:
            return 1
        
        # 如果已经计算过，直接返回
        if self._optimal_batch_size is not None:
            return self._optimal_batch_size
        
        allocated, total, free = self._get_gpu_memory_info()
        
        # 估算每个距离计算需要的显存（保守估计）
        # 距离矩阵: batch_size * n * 4 bytes (float32)
        # embeddings: n * d * 4 bytes
        # min_distances: n * 4 bytes
        bytes_per_float = 4
        
        # 保守估计：只使用可用显存的30%用于批量计算
        available_memory = free * 0.3 if free > 0 else total * 0.1
        
        # 计算批量大小
        # 需要显存 = batch_size * n * 4 + n * d * 4 + n * 4
        # available_memory >= batch_size * n * 4 + (n * d + n) * 4
        # batch_size <= (available_memory / 4 - n * d - n) / n
        
        if available_memory > 0 and n > 0:
            base_memory = (n * d + n) * bytes_per_float
            batch_memory_per_item = n * bytes_per_float
            
            if available_memory > base_memory:
                estimated_batch = int((available_memory - base_memory) / batch_memory_per_item)
                # 限制在合理范围内
                estimated_batch = max(1, min(estimated_batch, self.max_batch_size, n // 10))
            else:
                estimated_batch = 1
        else:
            estimated_batch = 1
        
        # 根据数据规模动态调整
        if n < 10000:
            estimated_batch = min(estimated_batch, 32)
        elif n < 100000:
            estimated_batch = min(estimated_batch, 128)
        else:
            estimated_batch = min(estimated_batch, 512)
        
        self._optimal_batch_size = estimated_batch
        
        logger.info(
            f"Adaptive batch size: {estimated_batch} "
            f"(GPU memory: {allocated/1024**3:.2f}GB/{total/1024**3:.2f}GB, "
            f"available: {free/1024**3:.2f}GB, n={n}, d={d})"
        )
        
        return estimated_batch
    
    def _maybe_reduce_dim(self, embeddings: Tensor) -> Tensor:
        """可选降维"""
        if self.reduce_dim is None or self.reduce_dim >= embeddings.shape[1]:
            return embeddings
        
        if self.original_dim is None:
            self.original_dim = embeddings.shape[1]
        
        if self.projector is None:
            logger.info(f"Random projection: {embeddings.shape[1]} -> {self.reduce_dim}")
            self.projector = RandomProjection(embeddings.shape[1], self.reduce_dim)
            self.projector.fit(embeddings)
        
        return self.projector.transform(embeddings)
    
    def add_embeddings(self, embeddings: Tensor, image_names: Optional[List[str]] = None) -> None:
        """Add new embeddings to buffer"""
        if image_names is None:
            image_names = [f"unknown_{i}" for i in range(embeddings.shape[0])]
        
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        embeddings = self._maybe_reduce_dim(embeddings)
        
        if self.embeddings is None:
            self.embeddings = embeddings
            self.image_names = image_names.copy()
        elif self.embeddings.shape[0] < self.buffer_size:
            self.embeddings = torch.vstack([self.embeddings, embeddings])
            self.image_names.extend(image_names)
        else:
            logger.warning(f"Buffer full ({self.embeddings.shape[0]}). Sampling...")
            self.sample()
            if self.embeddings.shape[0] + embeddings.shape[0] <= self.buffer_size:
                self.embeddings = torch.vstack([self.embeddings, embeddings])
                self.image_names.extend(image_names)
            else:
                temp_embeddings = torch.vstack([self.embeddings, embeddings])
                temp_image_names = self.image_names + image_names
                sampled_indices = self._k_center_greedy_sample_indices(
                    temp_embeddings, self.num_centers / temp_embeddings.shape[0]
                )
                self.embeddings = temp_embeddings[sampled_indices]
                self.image_names = [temp_image_names[i] for i in sampled_indices]
    
    def sample(self, sampling_ratio: Optional[float] = None) -> Tensor:
        """Sample from embeddings using K-Center Greedy"""
        if self.embeddings is None:
            raise ValueError("No embeddings to sample")
        
        if sampling_ratio is None:
            sampling_ratio = self.num_centers / self.embeddings.shape[0]
        sampling_ratio = max(0.0, min(1.0, sampling_ratio))
        
        if self.embeddings.shape[0] <= self.num_centers:
            return self.embeddings
        
        sampled_indices = self._k_center_greedy_sample_indices(self.embeddings, sampling_ratio)
        self.embeddings = self.embeddings[sampled_indices]
        self.image_names = [self.image_names[i] for i in sampled_indices]
        
        logger.info(f"Sampled {self.embeddings.shape[0]} embeddings")
        return self.embeddings
    
    def _k_center_greedy_sample_indices(self, embeddings: Tensor, sampling_ratio: float) -> List[int]:
        """K-Center Greedy sampling with incremental distance update"""
        # 确保 embeddings 在正确的设备上
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        num_samples = max(1, min(int(embeddings.shape[0] * sampling_ratio), embeddings.shape[0]))
        
        if num_samples >= embeddings.shape[0]:
            return list(range(embeddings.shape[0]))
        
        n = embeddings.shape[0]
        
        logger.info(
            f"Start sampling: from {n} samples to select {num_samples} centers "
            f"(sampling rate: {sampling_ratio:.2%}, device: {embeddings.device})"
        )
        
        # 超大规模数据使用随机采样
        if n > 1000000 and num_samples > 100000:
            logger.warning(f"Data size too large (N={n}), using random sampling")
            np.random.seed(42)
            return np.random.choice(n, size=num_samples, replace=False).tolist()
        
        # FAISS 加速（优先使用 GPU 版本）
        if self.use_faiss and n > 10000:
            if self.use_faiss_gpu:
                return self._k_center_greedy_faiss_gpu(embeddings, num_samples)
            else:
                return self._k_center_greedy_faiss(embeddings, num_samples)
        
        return self._k_center_greedy_optimized(embeddings, num_samples)
    
    def _k_center_greedy_optimized(self, embeddings: Tensor, num_samples: int) -> List[int]:
        """优化的 K-Center Greedy (增量式距离更新，支持 GPU 加速和批量处理)"""
        n, d = embeddings.shape
        device = embeddings.device
        
        # 确保在正确的设备上
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        # 计算自适应批量大小
        batch_size = self._estimate_optimal_batch_size(n, d)
        
        np.random.seed(42)
        first_idx = np.random.randint(0, n)
        selected_indices = [first_idx]
        
        # 初始化：计算所有点到第一个中心的距离（在 GPU 上）
        min_distances = torch.cdist(embeddings, embeddings[first_idx:first_idx+1]).squeeze(1)
        min_distances[first_idx] = -float('inf')
        
        # 使用 tqdm 进度条
        desc = f"K-Center Greedy Sample (GPU accelerated, batch={batch_size})" if self.use_gpu else "K-Center Greedy Sample (CPU)"
        pbar = tqdm(
            range(1, num_samples),
            desc=desc,
            unit="sample",
            ncols=100
        )
        
        # 优化的批量处理：使用候选池批量计算距离，提高GPU利用率
        if batch_size > 1 and self.use_gpu:
            for i in pbar:
                # 找到当前最远的 batch_size 个候选点
                _, top_indices = torch.topk(min_distances, min(batch_size, n - len(selected_indices)), largest=True)
                candidate_indices = top_indices.tolist()
                
                # 批量计算这些候选点到所有点的距离 [n, batch_size]
                candidate_embeddings = embeddings[candidate_indices]
                batch_distances = torch.cdist(embeddings, candidate_embeddings)
                
                # 选择最远的候选点（第一个，即最远的）
                best_candidate = candidate_indices[0]
                new_distances = batch_distances[:, 0]
                
                # 更新最小距离（增量式更新）
                min_distances = torch.minimum(min_distances, new_distances)
                min_distances[best_candidate] = -float('inf')
                selected_indices.append(best_candidate)
                
                # 更新进度条描述
                pbar.set_postfix({
                    'selected': f"{i + 1}/{num_samples}",
                    'progress': f"{(i + 1) / num_samples * 100:.1f}%",
                    'device': str(device),
                    'batch': len(candidate_indices)
                })
        else:
            # 单点模式：每次选择一个点（原始算法）
            for i in pbar:
                # 找到距离最远的点（已并行化）
                farthest_idx = torch.argmax(min_distances).item()
                selected_indices.append(farthest_idx)
                
                # 计算新中心到所有点的距离（在 GPU 上并行计算）
                new_distances = torch.cdist(embeddings, embeddings[farthest_idx:farthest_idx+1]).squeeze(1)
                
                # 更新最小距离（增量式更新，避免重复计算）
                min_distances = torch.minimum(min_distances, new_distances)
                min_distances[farthest_idx] = -float('inf')
                
                # 更新进度条描述
                pbar.set_postfix({
                    'selected': f"{i + 1}/{num_samples}",
                    'progress': f"{(i + 1) / num_samples * 100:.1f}%",
                    'device': str(device)
                })
        
        pbar.close()
        logger.info(
            f"K-Center Greedy sample completed: selected {len(selected_indices)} centers "
            f"(device: {device}, batch size: {batch_size})"
        )
        
        return selected_indices
    
    def _k_center_greedy_faiss(self, embeddings: Tensor, num_samples: int) -> List[int]:
        """FAISS CPU 加速的 K-Center Greedy"""
        n, d = embeddings.shape
        # 转换为 CPU numpy 数组（FAISS CPU 需要）
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        
        index = faiss.IndexFlatL2(d)
        
        np.random.seed(42)
        first_idx = np.random.randint(0, n)
        selected_indices = [first_idx]
        index.add(embeddings_np[first_idx:first_idx+1])
        
        distances, _ = index.search(embeddings_np, 1)
        min_distances = distances.squeeze(1)
        min_distances[first_idx] = -np.inf
        
        # 使用 tqdm 进度条
        pbar = tqdm(
            range(1, num_samples),
            desc="K-Center Greedy sample (FAISS CPU)",
            unit="sample",
            ncols=100
        )
        
        for i in pbar:
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(int(farthest_idx))
            index.add(embeddings_np[farthest_idx:farthest_idx+1])
            
            distances, _ = index.search(embeddings_np, 1)
            min_distances = np.minimum(min_distances, distances.squeeze(1))
            min_distances[farthest_idx] = -np.inf
            
            # 更新进度条描述
            pbar.set_postfix({
                'selected': f"{i + 1}/{num_samples}",
                'progress': f"{(i + 1) / num_samples * 100:.1f}%"
            })
        
        pbar.close()
        logger.info(f"K-Center Greedy sample completed (FAISS CPU): selected {num_samples} centers")
        
        return selected_indices
    
    def _k_center_greedy_faiss_gpu(self, embeddings: Tensor, num_samples: int) -> List[int]:
        """FAISS GPU 加速的 K-Center Greedy"""
        n, d = embeddings.shape
        
        # 使用 FAISS GPU
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        
        np.random.seed(42)
        first_idx = np.random.randint(0, n)
        selected_indices = [first_idx]
        index.add(embeddings_np[first_idx:first_idx+1])
        
        distances, _ = index.search(embeddings_np, 1)
        min_distances = distances.squeeze(1)
        min_distances[first_idx] = -np.inf
        
        # 使用 tqdm 进度条
        pbar = tqdm(
            range(1, num_samples),
            desc="K-Center Greedy sample (FAISS GPU)",
            unit="sample",
            ncols=100
        )
        
        for i in pbar:
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(int(farthest_idx))
            index.add(embeddings_np[farthest_idx:farthest_idx+1])
            
            distances, _ = index.search(embeddings_np, 1)
            min_distances = np.minimum(min_distances, distances.squeeze(1))
            min_distances[farthest_idx] = -np.inf
            
            # 更新进度条描述
            pbar.set_postfix({
                'selected': f"{i + 1}/{num_samples}",
                'progress': f"{(i + 1) / num_samples * 100:.1f}%"
            })
        
        pbar.close()
        logger.info(f"K-Center Greedy sample completed (FAISS GPU): selected {num_samples} centers")
        
        return selected_indices
    
    def get_image_names(self) -> List[str]:
        return self.image_names.copy()
    
    def reset(self) -> None:
        self.embeddings = None
        self.image_names = []
    
    def get_embeddings(self) -> Optional[Tensor]:
        return self.embeddings
    
    def get_statistics(self) -> dict:
        return {
            'total_embeddings': self.embeddings.shape[0] if self.embeddings is not None else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'original_dim': self.original_dim,
            'reduce_dim': self.reduce_dim,
            'buffer_size': self.buffer_size,
            'num_centers': self.num_centers,
            'buffer_usage': self.embeddings.shape[0] / self.buffer_size if self.embeddings is not None else 0,
            'device': str(self.device),
            'use_gpu': self.use_gpu,
            'use_faiss_gpu': self.use_faiss_gpu,
        }
    
    def get_projector(self) -> Optional[RandomProjection]:
        return self.projector

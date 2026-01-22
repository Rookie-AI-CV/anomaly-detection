import logging
from typing import Optional
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class KCenterGreedyMemoryBank:
    """
    K-Center Greedy 
    
    Implement batch loading data strategy to avoid memory overflow.
    When buffer is full, automatically sample, then add new embeddings.
    """
    
    def __init__(
        self,
        num_centers: int = 50000,
        buffer_size: int = 2500000,
    ):
        """
        Initialize KCenterGreedyMemoryBank.
        Args:
            num_centers: Maximum number of samples in memory bank
            buffer_size: Maximum size of buffer (for accumulating embeddings)
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
        Add new embeddings to buffer.
        
        When buffer is full, automatically sample, then add new embeddings.
        
        Args:
            embeddings: New embeddings [N, D]
        """
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            if self.embeddings.shape[0] < self.buffer_size:
                # Buffer is not full, add directly
                self.embeddings = torch.vstack([self.embeddings, embeddings])
            else:
                # Buffer is full, sample before adding
                logger.warning(
                    f"Buffer full ({self.embeddings.shape[0]} >= {self.buffer_size}). "
                    "Sampling before adding new embeddings."
                )
                self.sample()
                # Add new embeddings
                if self.embeddings.shape[0] + embeddings.shape[0] <= self.buffer_size:
                    self.embeddings = torch.vstack([self.embeddings, embeddings])
                else:
                    # Still too large, merge and resample
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
        Sample from accumulated embeddings to build memory bank.
        
        Args:
            sampling_ratio: Sampling ratio (if None, use num_centers)
            
        Returns:
            Sampled embeddings [num_centers, D]
        """
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            raise ValueError("No embeddings to sample. Add embeddings first.")
        
        if sampling_ratio is None:
            sampling_ratio = self.num_centers / self.embeddings.shape[0]
        
        # Limit sampling ratio to [0, 1]
        sampling_ratio = max(0.0, min(1.0, sampling_ratio))
        
        # If number of embeddings is less than num_centers, return all embeddings
        if self.embeddings.shape[0] <= self.num_centers:
            logger.info(
                f"Embeddings ({self.embeddings.shape[0]}) <= num_centers ({self.num_centers}). "
                "Returning all embeddings."
            )
            return self.embeddings
        
        # Use K-Center Greedy sampling
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
        Use K-Center Greedy algorithm to sample.
        
        Args:
            embeddings: Input embeddings [N, D]
            sampling_ratio: Sampling ratio
            
        Returns:
            Sampled embeddings [M, D], where M = int(N * sampling_ratio)
        """
        num_samples = int(embeddings.shape[0] * sampling_ratio)
        num_samples = max(1, min(num_samples, embeddings.shape[0]))
        
        if num_samples >= embeddings.shape[0]:
            return embeddings
        
        # Use simple K-Center Greedy implementation
        # Select first sample (random or first)
        selected_indices = [0]
        distances = torch.cdist(embeddings, embeddings[0:1])[:, 0]
        
        # Iterate to select remaining samples
        for _ in range(1, num_samples):
            # Find the farthest point from the selected sample set
            min_distances = torch.min(
                torch.cdist(embeddings, embeddings[selected_indices]), dim=1
            )[0]
            # Select the farthest point
            farthest_idx = torch.argmax(min_distances).item()
            selected_indices.append(farthest_idx)
        
        return embeddings[selected_indices]
    
    def reset(self) -> None:
        """Reset memory bank buffer."""
        self.embeddings = None
        logger.info("Memory bank reset.")
    
    def get_embeddings(self) -> Optional[Tensor]:
        """
        Get current embedding buffer.
        
        Returns:
            Current embeddings or None
        """
        return self.embeddings
    
    def get_statistics(self) -> dict:
        """
        Get statistics.
        
        Returns:
            Statistics dictionary
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

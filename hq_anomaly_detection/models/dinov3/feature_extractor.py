"""
DINOv3 feature extractor.

Uses DINOv3 models from timm to extract image-level features (cls_token).
Supports local model paths, does not auto-download.
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Optional, Union
import torch
from torch import nn

# Set Hugging Face mirror endpoint (if not set)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm

# Try to import safetensors (for loading .safetensors format)
try:
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

DINOV3_MODEL_NAME_MAP = {
    "dinov3_small": "vit_small_patch16_dinov3",
    "dinov3_base": "vit_base_patch16_dinov3",
    "dinov3_large": "vit_large_patch16_dinov3",
    "dinov3_huge": "vit_huge_patch16_dinov3",
}

# DINOv3 model Hugging Face Hub repository mapping
DINOV3_HF_REPO_MAP = {
    "dinov3_small": "timm/vit_small_patch16_dinov3.lvd1689m",
    "dinov3_base": "timm/vit_base_patch16_dinov3.lvd1689m",
    "dinov3_large": "timm/vit_large_patch16_dinov3.lvd1689m",
    "dinov3_huge": "timm/vit_huge_patch16_dinov3.lvd1689m",
}


def _get_timm_model_name(model_name: str) -> str:
    """
    Get timm model name.
    
    Args:
        model_name: Model name (could be simplified name or full timm name)
        
    Returns:
        Timm model name
    """
    if model_name in DINOV3_MODEL_NAME_MAP:
        return DINOV3_MODEL_NAME_MAP[model_name]
    return model_name


class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3 feature extractor.
    
    Uses DINOv3 models from timm to extract image-level features.
    Only extracts cls_token as image-level features.
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_base",
        model_path: Optional[Union[str, Path]] = None,
        use_cls_token: bool = True,
        **kwargs
    ):
        """
        Initialize DINOv3 feature extractor.
        
        Args:
            model_name: DINOv3 model name (e.g., 'dinov3_base', 'dinov3_large')
            model_path: Local model weight path (if provided, don't download automatically)
            use_cls_token: Whether to use cls_token (image-level features), default True
            **kwargs: Other parameters
        """
        super().__init__()
        self.model_name = model_name
        # Skip downloading if model_path is a special marker (will load from checkpoint)
        if model_path == "__checkpoint__":
            self.model_path = None
            self._skip_download = True
        else:
            self.model_path = Path(model_path) if model_path else None
            self._skip_download = False
        self.use_cls_token = use_cls_token
        
        # Get timm model name (may need mapping)
        timm_model_name = _get_timm_model_name(model_name)
        
        # Load model
        if self._skip_download:
            # Skip downloading, will load from checkpoint later
            logger.info(f"Skipping model loading (will load from checkpoint): {timm_model_name}")
            
            # Try multiple possible model names for huge models
            possible_names = [timm_model_name]
            if 'huge' in model_name.lower() or 'huge' in timm_model_name.lower():
                possible_names.extend([
                    'vit_huge_patch16_dinov3',
                    'vit_huge_plus_patch16_dinov3',
                    'vit_huge_patch14_dinov3',
                ])
            
            # Try each possible model name
            backbone_model = None
            for name in possible_names:
                try:
                    backbone_model = timm.create_model(
                        model_name=name,
                        pretrained=False,
                        num_classes=0,
                        **kwargs
                    )
                    logger.info(f"Successfully created model with name: {name}")
                    break
                except Exception as e:
                    if name == possible_names[-1]:
                        raise RuntimeError(f"Failed to create model with any of the names: {possible_names}. Last error: {e}")
                    continue
            
            self.backbone_model = backbone_model
        elif self.model_path and self.model_path.exists():
            # Load model from local path
            logger.info(f"Loading DINOv3 model from local path: {self.model_path}")
            
            # Try to infer model name from path if timm_model_name doesn't work
            # Check if parent directory contains model name hints
            model_path_str = str(self.model_path)
            possible_names = [timm_model_name]
            
            # Try to extract model name from path (e.g., vit_huge_plus_patch16_dinov3)
            if 'vit_huge' in model_path_str.lower():
                possible_names.extend([
                    'vit_huge_patch16_dinov3',
                    'vit_huge_plus_patch16_dinov3',
                    'vit_huge_patch14_dinov3',
                ])
            
            # Try each possible model name
            backbone_model = None
            for name in possible_names:
                try:
                    backbone_model = timm.create_model(
                        model_name=name,
                        pretrained=False,
                        num_classes=0,
                        **kwargs
                    )
                    logger.info(f"Successfully created model with name: {name}")
                    break
                except Exception as e:
                    if name == possible_names[-1]:
                        raise RuntimeError(f"Failed to create model with any of the names: {possible_names}. Last error: {e}")
                    continue
            
            self.backbone_model = backbone_model
            # Load local weights (supports .pth, .pt, .safetensors formats)
            model_path_str = str(self.model_path)
            
            if model_path_str.endswith('.safetensors'):
                # Use safetensors to load
                if not SAFETENSORS_AVAILABLE:
                    raise ImportError(
                        "safetensors library is required to load .safetensors files. "
                        "Please install it with: pip install safetensors"
                    )
                logger.info("Loading weights from safetensors format...")
                state_dict = safe_load_file(model_path_str)
            else:
                # Use torch.load to load (.pth, .pt formats)
                logger.info("Loading weights from PyTorch format...")
                checkpoint = torch.load(self.model_path, map_location='cpu')
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            
            # Remove 'module.' prefix (if saved with DataParallel)
            state_dict = {
                k.replace('module.', ''): v
                for k, v in state_dict.items()
            }
            
            # Load weights
            missing_keys, unexpected_keys = self.backbone_model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5 only
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
        else:
            # Load from timm (will auto-download)
            logger.info(f"Loading DINOv3 model from timm: {timm_model_name} (original name: {model_name})")
            if self.model_path:
                logger.warning(
                    f"Model path {self.model_path} does not exist. "
                    "Will try to load from timm (may download)."
                )
            
            # Must load pretrained weights, add retry mechanism
            max_retries = 3
            retry_delay = 5  # seconds
            
            # Get Hugging Face repository information
            hf_repo = DINOV3_HF_REPO_MAP.get(model_name, f"timm/{timm_model_name}.lvd1689m")
            hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            hf_url = f"{hf_endpoint}/{hf_repo}"
            
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        f"Attempting to load pretrained weights (attempt {attempt}/{max_retries})..."
                    )
                    self.backbone_model = timm.create_model(
                        model_name=timm_model_name,
                        pretrained=True,
                        num_classes=0,  # Remove classification head
                        **kwargs
                    )
                    logger.info("Successfully loaded pretrained weights from timm")
                    break  # Success, exit retry loop
                except Exception as e:
                    last_error = e
                    error_msg = str(e)[:300]  # Limit error message length
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {error_msg}"
                    )
                    
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # All retries failed, output download link information
                        logger.error("=" * 80)
                        logger.error("Failed to download pretrained weights automatically!")
                        logger.error("=" * 80)
                        logger.error(f"Model: {model_name} ({timm_model_name})")
                        logger.error(f"Hugging Face Repository: {hf_repo}")
                        logger.error(f"Hugging Face URL: {hf_url}")
                        logger.error(f"HF_ENDPOINT: {hf_endpoint}")
                        logger.error("")
                        logger.error("You can manually download the weights using one of the following methods:")
                        logger.error("")
                        logger.error("Method 1: Use the download script:")
                        logger.error(f"  python scripts/download_dinov3_weights.py --model-name {model_name}")
                        logger.error("")
                        logger.error("Method 2: Use huggingface-cli:")
                        logger.error(f"  export HF_ENDPOINT={hf_endpoint}")
                        logger.error(f"  huggingface-cli download {hf_repo} --local-dir ./weights/{model_name}")
                        logger.error("")
                        logger.error("Method 3: Download from browser:")
                        logger.error(f"  Visit: {hf_url}")
                        logger.error(f"  Download 'pytorch_model.bin' file")
                        logger.error("")
                        logger.error("After downloading, set model_path in your config:")
                        logger.error(f"  model_path: ./weights/{model_name}/pytorch_model.bin")
                        logger.error("=" * 80)
                        logger.error("")
                        logger.error(
                            f"Failed to load pretrained weights after {max_retries} attempts. "
                            f"Last error: {type(e).__name__}: {error_msg}"
                        )
                        raise RuntimeError(
                            f"Failed to load pretrained weights for model '{timm_model_name}' "
                            f"after {max_retries} attempts. This is required for training.\n\n"
                            f"Download information:\n"
                            f"  - Hugging Face Repository: {hf_repo}\n"
                            f"  - Hugging Face URL: {hf_url}\n"
                            f"  - HF_ENDPOINT: {hf_endpoint}\n\n"
                            f"Please manually download the weights or check your network connection.\n"
                            f"See the logs above for detailed download instructions.\n\n"
                            f"Last error: {type(e).__name__}: {error_msg}"
                        ) from e
        
        # Set to evaluation mode
        self.backbone_model.eval()
        
        # Get embedding dimension
        self.embed_dim = self._get_embed_dim()
        
        logger.info(
            f"Initialized DINOv3FeatureExtractor: "
            f"model={model_name}, embed_dim={self.embed_dim}, "
            f"use_cls_token={use_cls_token}, "
            f"model_path={model_path if model_path else 'timm (auto-download)'}"
        )
    
    def _get_embed_dim(self) -> int:
        """Get embedding dimension"""
        if hasattr(self.backbone_model, 'embed_dim'):
            return self.backbone_model.embed_dim
        elif hasattr(self.backbone_model, 'num_features'):
            return self.backbone_model.num_features
        else:
            # Infer from model name
            if 'vitg' in self.model_name or 'giant' in self.model_name:
                return 1536
            elif 'vith' in self.model_name or 'huge' in self.model_name:
                return 1280
            elif 'vitl' in self.model_name or 'large' in self.model_name:
                return 1024
            elif 'vitb' in self.model_name or 'base' in self.model_name:
                return 768
            elif 'vits' in self.model_name or 'small' in self.model_name:
                return 384
            else:
                return 768  # Default value
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features.
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, D] (cls_token, image-level features)
        """
        with torch.no_grad():
            features = self.backbone_model.forward_features(input_tensor)
        
        # Extract cls_token
        if self.use_cls_token:
            cls_token = self._extract_cls_token(features)
            return cls_token
        else:
            # If not using cls_token, return all patch tokens (flattened)
            patch_tokens = self._extract_patch_tokens(features)
            return patch_tokens
    
    def _extract_cls_token(self, features: Any) -> torch.Tensor:
        """
        Extract cls_token (image-level features).
        
        Args:
            features: Raw features from DINO model (dict or tensor)
            
        Returns:
            cls_token features [B, D]
        """
        if isinstance(features, dict):
            # Try different possible key names
            cls_token = (
                features.get('x_norm_clstoken') or
                features.get('cls_token') or
                features.get('x_cls_token') or
                features.get('cls')
            )
            
            if cls_token is None:
                # Fallback: use mean of patch tokens
                patch_tokens = (
                    features.get('x_norm_patchtokens') or
                    features.get('patchtokens')
                )
                if patch_tokens is not None:
                    cls_token = patch_tokens.mean(dim=1)
                else:
                    raise ValueError(
                        f"Could not find cls_token in features. "
                        f"Available keys: {list(features.keys())}"
                    )
        else:
            # If features is a tensor
            if features.dim() == 3:
                # [B, num_tokens, D] -> take first token (cls_token)
                cls_token = features[:, 0]
            elif features.dim() == 2:
                # [B, D] -> already cls_token
                cls_token = features
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return cls_token
    
    def _extract_patch_tokens(self, features: Any) -> torch.Tensor:
        """
        Extract patch tokens (patch-level features).
        
        Args:
            features: Raw features from DINO model
            
        Returns:
            patch token features [B*H*W, D]
        """
        if isinstance(features, dict):
            patch_tokens = (
                features.get('x_norm_patchtokens') or
                features.get('patchtokens') or
                features.get('x_patchtokens')
            )
            
            if patch_tokens is None:
                # If only full sequence available, skip cls_token
                full_seq = features.get('x_norm', None)
                if full_seq is not None and full_seq.dim() == 3:
                    # Skip first token (cls_token)
                    patch_tokens = full_seq[:, 1:]
                else:
                    raise ValueError(
                        f"Could not find patch tokens in features. "
                        f"Available keys: {list(features.keys())}"
                    )
        else:
            # If features is a tensor [B, num_tokens, D]
            if features.dim() == 3:
                # Skip first token (cls_token)
                patch_tokens = features[:, 1:]
            else:
                raise ValueError(
                    f"Unexpected feature shape for patch tokens: {features.shape}"
                )
        
        # Flatten to [B*H*W, D]
        batch_size, num_patches, embed_dim = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(batch_size * num_patches, embed_dim)
        
        return patch_tokens

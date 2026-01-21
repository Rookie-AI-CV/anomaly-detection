"""
PatchCore training module.

Provides training functionality for PatchCore models using anomalib.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import shutil
import os
import cv2

from omegaconf import OmegaConf
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.callbacks import get_callbacks

try:
    from anomalib.loggers import get_experiment_logger
except ImportError:
    def get_experiment_logger(config):
        """Create experiment logger from config or return None."""
        try:
            from lightning.pytorch.loggers import TensorBoardLogger
            log_dir = config.project.path if hasattr(config, 'project') and hasattr(config.project, 'path') else "."
            return TensorBoardLogger(save_dir=log_dir, name="anomalib")
        except (ImportError, AttributeError):
            return None

logger = logging.getLogger(__name__)


def get_configurable_parameters(
    config_path: str,
) -> OmegaConf:
    """Load configuration parameters from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config = OmegaConf.load(config_path)
    return config


def detect_img_max_size(train_dir: str) -> int:
    """Detect maximum image size from training directory.
    
    Args:
        train_dir: Path to training directory
        
    Returns:
        Maximum image size
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']
    image_dimensions = []
    root = Path(train_dir)
    
    for filename in root.glob(r"**/*"):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_path = str(filename)
            image = cv2.imread(image_path)
            if image is not None:
                height, width, _ = image.shape
                image_dimensions.append(max(width, height))
            else:
                logger.warning(f"Error reading {filename}")
    
    if len(image_dimensions) == 0:
        logger.warning(f"No valid images found in {train_dir}, using default size 256")
        return 256
    
    max_size = max(image_dimensions)
    logger.info(f"Detected max image size: {max_size}")
    return max_size


def merge_config(
    data_path: str,
    config_path: Optional[str] = None,
    device_id: int = 0,
) -> OmegaConf:
    """Merge configuration for PatchCore training.
    
    Args:
        data_path: Path to training data directory
        config_path: Path to YAML config file (optional)
        device_id: GPU device ID
        
    Returns:
        Merged OmegaConf configuration
    """
    data_path = Path(data_path).absolute()
    
    # Load base config if provided
    if config_path and Path(config_path).exists():
        conf = OmegaConf.load(config_path)
        
        # Ensure model has class_path for anomalib
        if 'model' not in conf:
            conf.model = OmegaConf.create({})
        
        # If model has 'name' but no 'class_path', convert to class_path format
        if 'name' in conf.model and 'class_path' not in conf.model:
            model_name = conf.model.pop('name')
            if model_name == 'patchcore':
                conf.model.class_path = 'anomalib.models.Patchcore'
                # Move remaining model parameters to init_args if they exist
                if 'init_args' not in conf.model:
                    conf.model.init_args = OmegaConf.create({})
                # Copy existing parameters to init_args
                model_params = dict(conf.model)
                for key in ['class_path', 'init_args']:
                    model_params.pop(key, None)
                if model_params:
                    conf.model.init_args.update(model_params)
        
        # Ensure model has init_args if class_path exists
        if 'class_path' in conf.model and 'init_args' not in conf.model:
            conf.model.init_args = OmegaConf.create({})
    else:
        # Create minimal default config
        conf = OmegaConf.create({
            'data': {
                'class_path': 'anomalib.data.Folder',
                'init_args': {
                    'name': 'custom_dataset',
                    'root': str(data_path),
                    'normal_dir': 'train/good',
                    'train_batch_size': 32,
                    'eval_batch_size': 32,
                    'num_workers': 8,
                },
            },
            'model': {
                'class_path': 'anomalib.models.Patchcore',
                'init_args': {
                    'backbone': 'wide_resnet50_2',
                    'pre_trained': True,
                    'layers': ['layer2', 'layer3'],
                    'coreset_sampling_ratio': 0.01,
                    'num_neighbors': 9,
                },
            },
            'trainer': {
                'accelerator': 'gpu',
                'devices': [device_id],
                'enable_checkpointing': True,
                'max_epochs': 1,
                'num_sanity_val_steps': 0,  # Skip sanity check to avoid empty embedding store error
            },
            'project': {
                'path': str(data_path / 'output'),
            },
        })
    
    # Ensure data path is set
    if 'data' in conf and 'init_args' in conf.data:
        if 'root' not in conf.data.init_args:
            conf.data.init_args.root = str(data_path)
    
    # Check if the data path is a local custom path (not the default MVTec dataset path)
    is_custom_local_path = False
    actual_data_path = data_path
    
    if 'data' in conf and 'init_args' in conf.data and 'root' in conf.data.init_args:
        actual_data_path = Path(conf.data.init_args.root)
    
    # Check if it's not the default MVTecAD path
    default_mvtec_paths = ['./datasets/MVTecAD', 'datasets/MVTecAD', '~/datasets/MVTecAD']
    if str(actual_data_path) not in default_mvtec_paths and actual_data_path.exists():
        is_custom_local_path = True
        logger.info(f"Detected custom local data path: {actual_data_path}, will use Folder datamodule")
    
    # Ensure we use Folder datamodule for custom paths
    if is_custom_local_path:
        if conf.data.class_path != 'anomalib.data.Folder':
            conf.data.class_path = 'anomalib.data.Folder'
            logger.info(f"Changed data module to Folder for custom local path")
        
        # Remove unsupported parameters for Folder
        if 'init_args' in conf.data:
            # Folder does not accept 'image_size' or 'task' parameters
            if 'image_size' in conf.data.init_args:
                del conf.data.init_args['image_size']
                logger.info("Removed unsupported 'image_size' parameter for Folder datamodule")
            if 'task' in conf.data.init_args:
                del conf.data.init_args['task']
                logger.info("Removed unsupported 'task' parameter for Folder datamodule")
            
            # Ensure required parameters are set
            if 'name' not in conf.data.init_args:
                conf.data.init_args.name = 'custom_dataset'
    
    # Detect image size if not set
    train_dir = actual_data_path / 'train'
    if train_dir.exists() and 'init_args' not in conf.data.get('init_args', {}):
        max_img_size = detect_img_max_size(str(train_dir))
        logger.info(f"Detected max image size: {max_img_size}")
    
    # Ensure project path exists
    if 'project' in conf and 'path' in conf.project:
        project_path = Path(conf.project.path)
        project_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Project output path: {project_path}")
    
    return conf


def train_patchcore(
    data_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    device_id: int = 0,
) -> None:
    """Train PatchCore model.
    
    Args:
        data_path: Path to training data directory
        config_path: Path to YAML config file (optional)
        device_id: GPU device ID
    """
    logger.info("Training PatchCore model")
    
    data_path = Path(data_path).absolute()
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    if config_path:
        logger.info(f"Using config file: {config_path}")
    else:
        logger.info("Using default config")
    
    # Merge configuration
    config = merge_config(
        data_path=str(data_path),
        config_path=str(config_path) if config_path else None,
        device_id=device_id,
    )
    
    # Ensure model.class_path exists (final check)
    if 'model' not in config:
        config.model = OmegaConf.create({})
    
    if 'class_path' not in config.model:
        config.model.class_path = 'anomalib.models.Patchcore'
        logger.info("Added missing model.class_path: anomalib.models.Patchcore")
    
    if 'init_args' not in config.model:
        config.model.init_args = OmegaConf.create({})
    
    # Log final model config for debugging
    logger.debug(f"Final model config: {config.model}")
    
    # Get data module
    datamodule = get_datamodule(config)
    
    # Get model (pass config.model, not entire config)
    model = get_model(config.model)
    
    # Get experiment logger
    experiment_logger = get_experiment_logger(config)
    
    # Get callbacks
    callbacks = get_callbacks(config)
    
    # Create trainer
    trainer_args = OmegaConf.to_container(config.trainer, resolve=True)
    if not isinstance(trainer_args, dict):
        trainer_args = dict(trainer_args)
    
    # Filter unsupported trainer parameters
    import inspect
    trainer_sig = inspect.signature(Trainer.__init__)
    supported_params = set(trainer_sig.parameters.keys())
    
    # Remove unsupported parameters
    unsupported_params = [
        'track_grad_norm',
        'auto_lr_find',
        'auto_scale_batch_size',
        'replace_sampler_ddp',
        'progress_bar_refresh_rate',
    ]
    for param in unsupported_params:
        trainer_args.pop(param, None)
    
    # Filter parameters not in Trainer's signature
    filtered_args = {k: v for k, v in trainer_args.items() if k in supported_params}
    
    # Remove None values for parameters that don't accept None
    none_invalid_params = ['strategy', 'accelerator']
    for param in none_invalid_params:
        if param in filtered_args and filtered_args[param] is None:
            filtered_args.pop(param, None)
    
    # Remove 'callbacks' and 'logger' if present since we're passing them separately
    filtered_args.pop('callbacks', None)
    filtered_args.pop('logger', None)
    
    trainer = Trainer(**filtered_args, logger=experiment_logger, callbacks=callbacks)
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    logger.info("Training completed!")

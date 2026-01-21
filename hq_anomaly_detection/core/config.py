"""
Configuration management module.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize config from dictionary."""
        self._config = config_dict or {}
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load config from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        return cls(config_dict)
    
    def save(self, save_path: Union[str, Path]):
        """Save config to YAML file."""
        save_path = Path(save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Saved config to {save_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key (e.g., 'model.name')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set config value by dot-separated key (e.g., 'model.name')."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config.copy()

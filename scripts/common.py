"""脚本公共初始化"""

import os
import sys
from pathlib import Path

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, str(Path(__file__).parent))

from env_config import load_env
load_env()

#!/usr/bin/env python3
"""
Environment variables configuration for training scripts.
"""

import os
from pathlib import Path


def load_env():
    """Load environment variables from configuration."""
    # Environment variables for proxy and certificates
    env_vars = {
        "no_proxy": "localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,hf-mirror.com,mirrors.aliyun.com,pypi.org,pypi.python.org,*.pypi.org",
        "http_proxy": "http://172.29.51.4:12798",
        "https_proxy": "http://172.29.51.4:12798",
        "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    }
    
    # Set Hugging Face mirror endpoint (if not already set)
    if "HF_ENDPOINT" not in os.environ:
        env_vars["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars


def setup_env():
    """Setup environment variables (alias for load_env)."""
    return load_env()


if __name__ == "__main__":
    load_env()
    print("Environment variables loaded:")
    for key, value in os.environ.items():
        if key in ["no_proxy", "http_proxy", "https_proxy", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "HF_ENDPOINT"]:
            print(f"  {key}={value}")

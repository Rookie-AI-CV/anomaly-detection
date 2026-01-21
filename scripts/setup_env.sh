#!/bin/bash
# Environment variables setup for training scripts

export no_proxy=localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,hf-mirror.com,mirrors.aliyun.com,pypi.org,pypi.python.org,*.pypi.org
export http_proxy=http://172.29.51.4:12798
export https_proxy=http://172.29.51.4:12798
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_ENDPOINT=https://hf-mirror.com
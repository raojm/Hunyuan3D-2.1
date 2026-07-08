#!/bin/bash
# 统一通过 HF 镜像下载所有模型
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface

CACHE_DIR=~/.cache/hy3dgen/tencent/Hunyuan3D-2.1

echo "=== 1/4 Downloading Shape model (hunyuan3d-dit-v2-1, ~7GB) ==="
huggingface-cli download tencent/Hunyuan3D-2.1 \
  --include "hunyuan3d-dit-v2-1/*" \
  --local-dir $CACHE_DIR

echo "=== 2/4 Downloading Paint model (hunyuan3d-paintpbr-v2-1, ~4GB) ==="
huggingface-cli download tencent/Hunyuan3D-2.1 \
  --include "hunyuan3d-paintpbr-v2-1/*" \
  --local-dir $CACHE_DIR

echo "=== 3/4 Downloading VAE model (hunyuan3d-vae-v2-1, ~626MB) ==="
huggingface-cli download tencent/Hunyuan3D-2.1 \
  --include "hunyuan3d-vae-v2-1/*" \
  --local-dir $CACHE_DIR

echo "=== 4/4 Downloading DINOv2 (facebook/dinov2-giant, ~1.4GB) ==="
huggingface-cli download facebook/dinov2-giant --local-dir ~/.cache/huggingface/hub/models--facebook--dinov2-giant

echo "=== All models downloaded! ==="
echo "Cache location: $CACHE_DIR"

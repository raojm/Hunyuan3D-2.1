#!/bin/bash
# 统一通过 HF 镜像下载所有模型到标准 HF 缓存
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface

CACHE_DIR=~/.cache/hy3dgen/tencent/Hunyuan3D-2.1

echo "=== 1/5 Shape model (hunyuan3d-dit-v2-1, ~7GB) ==="
huggingface-cli download tencent/Hunyuan3D-2.1 \
  --include "hunyuan3d-dit-v2-1/*" \
  --local-dir $CACHE_DIR

echo "=== 2/5 Paint model (hunyuan3d-paintpbr-v2-1, ~4GB) ==="
huggingface-cli download tencent/Hunyuan3D-2.1 \
  --include "hunyuan3d-paintpbr-v2-1/*" \
  --local-dir $CACHE_DIR

echo "=== 3/5 VAE model (hunyuan3d-vae-v2-1, ~626MB) ==="
huggingface-cli download tencent/Hunyuan3D-2.1 \
  --include "hunyuan3d-vae-v2-1/*" \
  --local-dir $CACHE_DIR

echo "=== 4/5 DINOv2 (facebook/dinov2-giant, ~1.4GB) ==="
python3 -c "from transformers import AutoModel, AutoImageProcessor; AutoModel.from_pretrained('facebook/dinov2-giant'); AutoImageProcessor.from_pretrained('facebook/dinov2-giant'); print('DINOv2 cached OK')"

echo "=== 5/5 Stable Diffusion 2.1 (stabilityai/stable-diffusion-2-1, ~5GB) ==="
python3 -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1'); print('SD 2.1 cached OK')"

echo ""
echo "=== ALL MODELS DOWNLOADED ==="
echo "Run with: HF_HUB_OFFLINE=1 python demo.py"

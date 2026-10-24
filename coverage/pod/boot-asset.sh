#!/bin/bash
# Pod boot for a new character (coverage/asset/README.md): the file/job server, then the stack and
# the weights for all three GPU steps while the inputs upload: Qwen-Image-Edit-2509 (style reference
# and anchors), Wan2.1 VACE 14B + the lightx2v LoRA (the film), and torch for shader_torch.py.
mkdir -p /work/logs && cd /work
echo "$SERVER_PY" | base64 -d > /work/server.py
nohup python3 /work/server.py > /work/logs/server.log 2>&1 &
exec > /work/logs/setup.log 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | head -2; df -h / | tail -1
apt-get update -qq && apt-get install -y -qq zstd ffmpeg > /dev/null
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q diffusers==0.41.0 transformers accelerate peft ftfy sentencepiece imageio imageio-ffmpeg opencv-python-headless hf_transfer huggingface_hub 2>&1 | tail -3
python3 -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'diffusers', diffusers.__version__)"
date -u +%T; python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download('Qwen/Qwen-Image-Edit-2509'); print('qwen edit ready', flush=True)
hf_hub_download('Kijai/WanVideo_comfy', 'Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors')
snapshot_download('Wan-AI/Wan2.1-VACE-14B-diffusers'); print('vace ready')"
date -u +%T; echo "== setup done"

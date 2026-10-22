#!/bin/bash
# Pod boot: start the file/job server, then install the stack and fetch the weights while we upload.
mkdir -p /work/logs && cd /work
echo "$SERVER_PY" | base64 -d > /work/server.py
nohup python3 /work/server.py > /work/logs/server.log 2>&1 &
exec > /work/logs/setup.log 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | head -2; df -h /work | tail -1
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q diffusers==0.41.0 transformers accelerate peft ftfy sentencepiece imageio imageio-ffmpeg hf_transfer huggingface_hub 2>&1 | tail -3
python3 -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__)"
date -u +%T; python3 -c "
from huggingface_hub import snapshot_download, hf_hub_download
hf_hub_download('Kijai/WanVideo_comfy', 'Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors')
snapshot_download('Wan-AI/Wan2.1-VACE-14B-diffusers')
print('weights ready')"
date -u +%T; echo "== setup done"

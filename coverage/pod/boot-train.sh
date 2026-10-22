#!/bin/bash
# Training pod boot: start the file/job server, check the stack, pre-fetch the VGG weights.
mkdir -p /work/logs && cd /work
echo "$SERVER_PY" | base64 -d > /work/server.py
nohup python3 /work/server.py > /work/logs/server.log 2>&1 &
exec > /work/logs/setup.log 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch, torchvision; print('torch', torch.__version__, 'vision', torchvision.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import torchvision; torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1); print('vgg ready')"
echo "== setup done"

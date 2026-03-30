#!/usr/bin/env bash
# One-time script to download all model weights to /workspace/models
# Run via SSH after first pod start: bash /app/scripts/download_models.sh
set -e

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$MODEL_DIR"
mkdir -p "$HF_HOME"

echo "Downloading Real-ESRGAN weights..."
python3 -c "
from basicsr.utils.download_util import load_file_from_url
urls = {
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'RealESRNet_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
}
import os
for name, url in urls.items():
    dest = os.path.join('$MODEL_DIR', name)
    if not os.path.exists(dest):
        load_file_from_url(url, model_dir='$MODEL_DIR', file_name=name)
        print(f'Downloaded {name}')
    else:
        print(f'Skipping {name} (already exists)')
"

echo "Downloading NAFNet weights..."
python3 -c "
import os
from huggingface_hub import hf_hub_download
dest = os.path.join('$MODEL_DIR', 'nafnet_reds.pth')
if not os.path.exists(dest):
    path = hf_hub_download('megvii-nanjing/NAFNet', filename='NAFNet-REDS-width64.pth', cache_dir='$HF_HOME')
    import shutil; shutil.copy(path, dest)
    print('Downloaded nafnet_reds.pth')
else:
    print('Skipping nafnet_reds.pth (already exists)')
"

echo "Downloading BiRefNet (via transformers — done at first inference)..."
echo "Downloading LaMa (via simple-lama-inpainting — done at first inference)..."

echo "Downloading InsightFace buffalo_l..."
python3 -c "
import insightface, os
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='$MODEL_DIR', providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1)
print('buffalo_l ready')
"

echo "Downloading inswapper_128.onnx..."
python3 -c "
import os
from huggingface_hub import hf_hub_download
dest = os.path.join('$MODEL_DIR', 'inswapper_128.onnx')
if not os.path.exists(dest):
    path = hf_hub_download('deepinsight/inswapper', filename='inswapper_128.onnx', cache_dir='$HF_HOME')
    import shutil; shutil.copy(path, dest)
    print('Downloaded inswapper_128.onnx')
else:
    print('Skipping inswapper_128.onnx (already exists)')
"

echo "All models downloaded to $MODEL_DIR"

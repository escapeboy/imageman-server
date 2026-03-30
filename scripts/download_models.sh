#!/usr/bin/env bash
# Download all model weights to /workspace/models
set -e

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$MODEL_DIR"
mkdir -p "$HF_HOME"

echo "Downloading Real-ESRGAN weights..."
python3 -c "
import os, urllib.request

MODEL_DIR = '$MODEL_DIR'
urls = {
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
}
for name, url in urls.items():
    dest = os.path.join(MODEL_DIR, name)
    if not os.path.exists(dest):
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(url, dest)
        print(f'Downloaded {name}')
    else:
        print(f'Skipping {name} (already exists)')
"

echo "Downloading NAFNet weights (optional — skip if unavailable)..."
python3 -c "
import os, urllib.request

MODEL_DIR = '$MODEL_DIR'
dest = os.path.join(MODEL_DIR, 'nafnet_reds.pth')
if os.path.exists(dest):
    print('Skipping nafnet_reds.pth (already exists)')
else:
    # Direct download from Google Drive via gdown or HuggingFace
    try:
        import subprocess, sys
        # Try HuggingFace first (Linaqruf hosts a copy)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
        import gdown
        file_id = '14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X'
        gdown.download(id=file_id, output=dest, quiet=False)
        print('Downloaded nafnet_reds.pth via gdown')
    except Exception as e:
        print(f'Warning: NAFNet download failed ({e}) — deblur will be unavailable')
" || true

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

echo "BiRefNet and LaMa: auto-downloaded at first inference."
echo "All model downloads complete."

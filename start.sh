#!/bin/bash
set -e

WORKSPACE=/workspace
REPO_URL="https://github.com/escapeboy/imageman-server.git"
APP_DIR="$WORKSPACE/app_src"
MODEL_DIR="$WORKSPACE/models"

echo "=== ImageMan Server Startup ==="

# Clone or update the repo
if [ -d "$APP_DIR/.git" ]; then
    echo ">> Updating repo..."
    cd "$APP_DIR" && git pull
else
    echo ">> Cloning repo..."
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# Install Python dependencies
echo ">> Installing requirements..."
pip install -q -r "$APP_DIR/requirements.txt"

# Patch basicsr for torchvision >= 0.16 compatibility
echo ">> Patching basicsr..."
python -c "
import os, site
for sp in site.getsitepackages():
    f = os.path.join(sp, 'basicsr/data/degradations.py')
    if os.path.exists(f):
        with open(f) as r: content = r.read()
        if 'functional_tensor' in content:
            content = content.replace(
                'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                'from torchvision.transforms.functional import rgb_to_grayscale'
            )
            with open(f, 'w') as w: w.write(content)
            print('Patched basicsr/data/degradations.py')
        break
"

# Download models (skip if already present)
echo ">> Checking models..."
export MODEL_DIR
bash "$APP_DIR/scripts/download_models.sh"

# Start the server
echo ">> Starting FastAPI server on port 8000..."
cd "$APP_DIR"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

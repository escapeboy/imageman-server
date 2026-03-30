import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_esrgan(model_name: str, scale: int) -> RealESRGANer:
    key = f"esrgan_{model_name}"
    cached = registry.get(key)
    if cached:
        return cached
    net = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale
    )
    path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=path,
        model=net,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    registry.put(key, upsampler)
    return upsampler

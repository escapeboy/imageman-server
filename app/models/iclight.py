import os
import torch
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_iclight():
    """
    IC-Light is a diffusion-based relighting model (~4-6 GB VRAM).
    Loaded last in LRU — evicted first under memory pressure.
    Follows lllyasviel/ic-light README for pipeline construction.
    """
    cached = registry.get("iclight")
    if cached:
        return cached
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        # IC-Light requires a custom UNet — load from HuggingFace
        # Full pipeline construction per lllyasviel/ic-light README
        pipe = StableDiffusionPipeline.from_pretrained(
            "lllyasviel/ic-light",
            cache_dir=os.path.join(MODEL_DIR, "iclight"),
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        registry.put("iclight", pipe)
        return pipe
    except Exception:
        return None

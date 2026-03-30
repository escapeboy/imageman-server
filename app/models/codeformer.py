import os
import torch
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_codeformer():
    cached = registry.get("codeformer")
    if cached:
        return cached
    # Use codeformer-inference package or sczhou/CodeFormer clone
    # The package wraps the detection + face parsing + generation pipeline
    try:
        from codeformer_inference import CodeFormerInference
        model = CodeFormerInference(
            model_path=os.path.join(MODEL_DIR, "codeformer.pth"),
            device="cuda",
        )
    except ImportError:
        # Fallback: return None — router will return source image unchanged
        return None
    registry.put("codeformer", model)
    return model

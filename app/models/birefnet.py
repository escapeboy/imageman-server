import os
import torch
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_birefnet():
    cached = registry.get("birefnet")
    if cached:
        return cached
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        cache_dir=os.path.join(MODEL_DIR, "birefnet"),
        trust_remote_code=True,
    )
    model.train(False)
    model.cuda().half()
    registry.put("birefnet", model)
    return model

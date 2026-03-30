import os
import torch
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_nafnet():
    cached = registry.get("nafnet")
    if cached:
        return cached
    model_path = os.path.join(MODEL_DIR, "nafnet_reds.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NAFNet model not found at {model_path}.")
    from app.models.nafnet_arch import NAFNet
    model = NAFNet(
        img_channel=3,
        width=64,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    state = torch.load(model_path, map_location="cpu")
    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "params" in state:
            weights = state["params"]
        elif "state_dict" in state:
            weights = state["state_dict"]
        elif "model" in state:
            weights = state["model"]
        else:
            weights = state
    else:
        weights = state
    model.load_state_dict(weights, strict=False)
    model.train(False)
    model.cuda()   # fp32 — fp16 causes overflow with NAFNet
    registry.put("nafnet", model)
    return model

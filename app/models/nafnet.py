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
        raise FileNotFoundError(f"NAFNet model not found at {model_path}. Deploy with NAFNet weights to use deblur.")
    from app.models.nafnet_arch import NAFNet
    model = NAFNet(
        img_channel=3,
        width=64,
        enc_blks=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blks=[1, 1, 1, 1],
    )
    state = torch.load(model_path, map_location="cuda")
    model.load_state_dict(state["params"])
    model.train(False)
    model.cuda().half()
    registry.put("nafnet", model)
    return model

import os
import torch
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_nafnet():
    cached = registry.get("nafnet")
    if cached:
        return cached
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3,
        width=64,
        enc_blks=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blks=[1, 1, 1, 1],
    )
    state = torch.load(
        os.path.join(MODEL_DIR, "nafnet_reds.pth"),
        map_location="cuda"
    )
    model.load_state_dict(state["params"])
    model.train(False)
    model.cuda().half()
    registry.put("nafnet", model)
    return model

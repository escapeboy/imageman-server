import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

from app.models.nafnet import get_nafnet
from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class DeblurRequest(BaseModel):
    source_image: str


@router.post("/deblur")
def deblur(req: DeblurRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        w, h = img.size
        # Pad to multiples of 16 for NAFNet
        pw = ((w + 15) // 16) * 16
        ph = ((h + 15) // 16) * 16
        padded = PILImage.new("RGB", (pw, ph))
        padded.paste(img, (0, 0))

        inp = torch.from_numpy(np.array(padded)).float() / 255.0
        inp = inp.permute(2, 0, 1).unsqueeze(0).cuda()  # fp32

        model = get_nafnet()
        with torch.no_grad():
            out = model(inp)

        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255).clip(0, 255).astype(np.uint8)
        result = PILImage.fromarray(out).crop((0, 0, w, h))
        return {"result_image": image_to_b64(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

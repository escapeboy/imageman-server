import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

from app.models.esrgan import get_esrgan
from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class UpscaleRequest(BaseModel):
    source_image: str
    model: str = "RealESRGAN_x4plus"
    scale: int = 4


@router.post("/upscale")
def upscale(req: UpscaleRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        sr = get_esrgan(req.model, req.scale)
        output, _ = sr.enhance(np.array(img), outscale=req.scale)
        return {"result_image": image_to_b64(PILImage.fromarray(output))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/face-enhance")
def face_enhance(req: UpscaleRequest):
    watchdog.ping()
    req.model = "RealESRGAN_x4plus"
    req.scale = 2
    return upscale(req)

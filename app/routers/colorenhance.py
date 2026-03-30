import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class ColorEnhanceRequest(BaseModel):
    source_image: str


@router.post("/color-enhance")
def color_enhance(req: ColorEnhanceRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return {"result_image": image_to_b64(PILImage.fromarray(result))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

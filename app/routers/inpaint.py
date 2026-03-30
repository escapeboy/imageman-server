from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.lama import get_lama
from app.utils import b64_to_image, b64_to_mask, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class InpaintRequest(BaseModel):
    source_image: str
    mask_image: str


@router.post("/inpaint")
def inpaint(req: InpaintRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        mask = b64_to_mask(req.mask_image)
        result = get_lama()(img, mask)
        return {"result_image": image_to_b64(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

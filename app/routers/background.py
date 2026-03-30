import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage, ImageFilter
from pydantic import BaseModel
from torchvision import transforms

from app.models.birefnet import get_birefnet
from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()

_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _get_alpha_mask(img: PILImage.Image) -> PILImage.Image:
    """Run BiRefNet inference and return a grayscale alpha mask (same size as img)."""
    orig_w, orig_h = img.size
    inp = _transform(img).unsqueeze(0).cuda().half()
    model = get_birefnet()
    with torch.no_grad():
        preds = model(inp)
    pred = preds[-1].sigmoid().squeeze().float().cpu().numpy()
    mask = PILImage.fromarray((pred * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), PILImage.LANCZOS
    )
    return mask


class BackgroundRequest(BaseModel):
    source_image: str


class BlurBackgroundRequest(BaseModel):
    source_image: str
    blur_radius: int = 15


@router.post("/remove-background")
def remove_background(req: BackgroundRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        mask = _get_alpha_mask(img)
        rgba = img.convert("RGBA")
        rgba.putalpha(mask)
        return {"result_image": image_to_b64(rgba, fmt="PNG")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blur-background")
def blur_background(req: BlurBackgroundRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        mask = _get_alpha_mask(img)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=req.blur_radius))
        composite = PILImage.composite(img, blurred, mask)
        return {"result_image": image_to_b64(composite)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

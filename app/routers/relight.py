from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.iclight import get_iclight
from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class RelightRequest(BaseModel):
    source_image: str
    bg_prompt: str = "indoor lighting, soft shadows"


@router.post("/relight")
def relight(req: RelightRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        pipe = get_iclight()
        if pipe is None:
            # IC-Light not available — return source unchanged
            return {"result_image": image_to_b64(img)}
        # IC-Light inference — exact call depends on pipeline version
        # See lllyasviel/ic-light README for full invocation
        result = pipe(
            prompt=req.bg_prompt,
            image=img,
            num_inference_steps=25,
        ).images[0]
        return {"result_image": image_to_b64(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

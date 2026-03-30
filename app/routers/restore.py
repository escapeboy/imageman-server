from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.codeformer import get_codeformer
from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class RestoreRequest(BaseModel):
    source_image: str
    prompt: str = ""  # accepted but ignored — CodeFormer is detection-based


@router.post("/restore")
def restore(req: RestoreRequest):
    watchdog.ping()
    try:
        img = b64_to_image(req.source_image)
        model = get_codeformer()
        if model is None:
            # codeformer-inference not installed — return source unchanged
            return {"result_image": image_to_b64(img)}
        result = model.restore(img)
        return {"result_image": image_to_b64(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

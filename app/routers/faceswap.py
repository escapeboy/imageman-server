import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image as PILImage
from pydantic import BaseModel

from app.models.faceswap import get_face_swapper
from app.utils import b64_to_image, image_to_b64
from app.watchdog import watchdog

router = APIRouter()


class FaceSwapRequest(BaseModel):
    source_image: str   # target image (body/background)
    target_image: str   # source face to swap in
    upscale: int = 2
    codeformer_fidelity: float = 0.7


@router.post("/face-swap")
def face_swap(req: FaceSwapRequest):
    watchdog.ping()
    try:
        target = b64_to_image(req.source_image)
        source = b64_to_image(req.target_image)

        face_app, swapper = get_face_swapper()

        target_cv = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2BGR)
        source_cv = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

        target_faces = face_app.get(target_cv)
        source_faces = face_app.get(source_cv)

        if not target_faces or not source_faces:
            raise ValueError("No face detected in source or target image")

        result = target_cv.copy()
        for face in target_faces:
            result = swapper.get(result, face, source_faces[0], paste_back=True)

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return {"result_image": image_to_b64(PILImage.fromarray(result_rgb))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

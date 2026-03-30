import os
from app.model_registry import registry

MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/models")


def get_face_swapper():
    cached = registry.get("faceswap")
    if cached:
        return cached
    import insightface
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(
        name="buffalo_l",
        root=MODEL_DIR,
        providers=["CUDAExecutionProvider"],
    )
    face_app.prepare(ctx_id=0)
    swapper = insightface.model_zoo.get_model(
        os.path.join(MODEL_DIR, "inswapper_128.onnx"),
        providers=["CUDAExecutionProvider"],
    )
    registry.put("faceswap", (face_app, swapper))
    return registry.get("faceswap")

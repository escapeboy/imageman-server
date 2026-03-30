from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.model_registry import registry
from app.watchdog import watchdog
from app.routers import upscale, deblur, background, inpaint, restore, relight, colorenhance, faceswap


@asynccontextmanager
async def lifespan(app: FastAPI):
    watchdog.start()
    yield
    watchdog.stop()


app = FastAPI(title="ImageMan AI Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(registry.loaded.keys())}


app.include_router(upscale.router)
app.include_router(deblur.router)
app.include_router(background.router)
app.include_router(inpaint.router)
app.include_router(restore.router)
app.include_router(relight.router)
app.include_router(colorenhance.router)
app.include_router(faceswap.router)

"""Dog Detection documentation site with live demo inference."""

import io
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from PIL import Image

# Optional YOLO import; the demo page will show a friendly error if it fails.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_AVAILABLE = False
    logger.warning(f"ultralytics not available: {exc}")


APP_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = Path("/home/biodiv/dogs_detection/runs/detect/DogDetection/train-30/weights/best.pt")
MODEL = None
MODEL_LOADED_AT = 0.0

app = FastAPI(title="Dog Detection Docs & Demo")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def load_model() -> "YOLO":
    """Lazy-load the YOLO model and cache it in memory."""
    global MODEL, MODEL_LOADED_AT
    if MODEL is None or not WEIGHTS_PATH.exists():
        if not WEIGHTS_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model weights not found at {WEIGHTS_PATH}",
            )
        if not ULTRALYTICS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Ultralytics is not installed; demo inference is unavailable.",
            )
        logger.info(f"Loading YOLO model from {WEIGHTS_PATH}")
        MODEL = YOLO(str(WEIGHTS_PATH))
        MODEL_LOADED_AT = time.time()
    return MODEL


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/demo", response_class=HTMLResponse)
def demo(request: Request):
    return templates.TemplateResponse(
        request,
        "demo.html",
        {
            "weights_path": str(WEIGHTS_PATH),
            "weights_exist": WEIGHTS_PATH.exists(),
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
        },
    )


@app.get("/cli", response_class=HTMLResponse)
def cli_reference(request: Request):
    return templates.TemplateResponse(request, "cli.html")


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
):
    """Run YOLO inference on an uploaded image and return the annotated image."""
    if not ULTRALYTICS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Ultralytics is not installed; demo inference is unavailable.",
        )
    if not WEIGHTS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not found at {WEIGHTS_PATH}",
        )

    try:
        contents = image.file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        logger.error(f"Invalid image upload: {exc}")
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    finally:
        image.file.close()

    model = load_model()
    start = time.time()
    results = model(
        np.asarray(pil_image),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    inference_time = (time.time() - start) * 1000

    annotated = results[0].plot(line_width=2, font_size=0.6)
    annotated_image = Image.fromarray(annotated)

    output = io.BytesIO()
    annotated_image.save(output, format="JPEG", quality=95)
    output.seek(0)

    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    logger.info(
        f"Demo inference: {num_detections} detections in {inference_time:.1f}ms "
        f"(conf={conf}, iou={iou}, imgsz={imgsz})"
    )

    return StreamingResponse(
        output,
        media_type="image/jpeg",
        headers={
            "X-Detections": str(num_detections),
            "X-Inference-Time-Ms": f"{inference_time:.1f}",
            "Content-Disposition": f'inline; filename="detection_{uuid.uuid4().hex[:8]}.jpg"',
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

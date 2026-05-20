#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import os
import tempfile
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import requests
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ----------------------------------------------------------------------------

app = FastAPI()

# --- Global Variables for Memory Management ---
MODEL_OBJ = None
WEIGHTS_PATH = None
MODEL_VERSION = None
IMAGE_DIR = None

LAST_ACTIVE_TIME = time.time()
IDLE_TIMEOUT_SECONDS = 300
# ----------------------------------------------------------------------------


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path/URL to the weights file',
                        type=str)
    parser.add_argument('-m',
                        '--model-version',
                        help='Name and model version',
                        type=str)
    parser.add_argument(
        '-d',
        '--image-dir',
        help='Optional: Local directory containing images to avoid downloading',
        type=str,
        default=None)
    parser.add_argument('-H',
                        '--host',
                        help='API host (default: 0.0.0.0)',
                        type=str,
                        default='0.0.0.0')
    parser.add_argument('-s',
                        '--port',
                        help='API port (default: 8000)',
                        type=int,
                        default=8000)
    return parser.parse_args()


# ----------------------------------------------------------------------------


class Task(BaseModel):
    task: dict
    project: Optional[int] = None


def load_model_lazy():
    """Loads the model into the GPU if it isn't already loaded."""
    global MODEL_OBJ
    if MODEL_OBJ is None:
        print(f"Cold start: Loading YOLO model from {WEIGHTS_PATH} to GPU...")
        MODEL_OBJ = YOLO(WEIGHTS_PATH)


def unload_model():
    """Deletes the model and forces the GPU to clear the VRAM cache."""
    global MODEL_OBJ
    if MODEL_OBJ is not None:
        print(
            f"Idle timeout ({IDLE_TIMEOUT_SECONDS}s) reached. Unloading YOLO model from GPU VRAM..."
        )
        del MODEL_OBJ
        MODEL_OBJ = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def memory_manager():
    """Background thread that checks for inactivity."""
    global LAST_ACTIVE_TIME
    while True:
        time.sleep(60)
        if MODEL_OBJ is not None and (time.time() -
                                      LAST_ACTIVE_TIME) > IDLE_TIMEOUT_SECONDS:
            unload_model()


def _yolo_to_ls(model, x: float, y: float, width: float, height: float,
                n: int) -> tuple:
    x = (x - width / 2) * 100
    y = (y - height / 2) * 100
    w = width * 100
    h = height * 100
    x, y, w, h = [float(i) for i in [x, y, w, h]]
    try:
        label = model.names[int(n)]
    except ValueError:
        label = n
    return x, y, w, h, label


def _pred_dict(model_version: str, x: float, y: float, w: float, h: float,
               label: str, score: float) -> dict:
    return {
        'type': 'rectanglelabels',
        'score': score,
        'value': {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'rectanglelabels': [label]
        },
        'to_name': 'image',
        'from_name': 'label',
        'model_version': model_version
    }


@app.post('/predict')
def predict_endpoint(task: Task):
    global LAST_ACTIVE_TIME
    LAST_ACTIVE_TIME = time.time()

    _task = task.task
    if not _task.get('project'):
        if task.project:
            _task['project'] = task.project
        else:
            raise HTTPException(
                404, 'Parameter `project` is required when the task does not '
                'contain a project id number!')
    task = _task

    load_model_lazy()
    model = MODEL_OBJ
    model_version = MODEL_VERSION

    image_url = task['data']['image']

    filename = os.path.basename(urlparse(image_url).path)

    local_image_path = None
    if IMAGE_DIR:
        potential_path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(potential_path):
            local_image_path = potential_path

    if local_image_path:
        model_preds = model(local_image_path)
    else:
        with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
            r = requests.get(image_url)
            if r.status_code == 200:
                f.write(r.content)
                f.flush()
            else:
                return JSONResponse(content=r.text, status_code=404)
            f.seek(0)
            model_preds = model(f.name)

    result = model_preds[0]
    results_list = []
    scores = []

    for box in result.boxes:
        x, y, w, h = box.xywhn[0].tolist()
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        scores.append(score)

        _result = _yolo_to_ls(model, x, y, w, h, cls_id)
        formatted_result = _pred_dict(model_version, *_result, score)
        results_list.append(formatted_result)

    if not results_list:
        return JSONResponse(status_code=200, content={})

    overall_score = sum(scores) / len(scores)

    pred = {
        'result': results_list,
        'score': overall_score,
        'model_version': model_version
    }

    return JSONResponse(status_code=200, content=pred)


# ----------------------------------------------------------------------------

if __name__ == '__main__':
    load_dotenv()
    args = opts()

    WEIGHTS_PATH = args.weights
    MODEL_VERSION = args.model_version
    IMAGE_DIR = args.image_dir

    threading.Thread(target=memory_manager, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)

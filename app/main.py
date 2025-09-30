from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from loguru import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from time import perf_counter

from .schemas import PredictResponse, BBox
from .inference import Detector
from .tracker import Tracker
from .utils import read_image_from_bytes
from .metrics import PREDICTIONS, ERRORS, LATENCY

app = FastAPI(title="Autonomous Vision AI", version="0.1.0")

detector = Detector()          # yolov8n.pt by default; swap to yolov8s.pt for better accuracy
tracker = Tracker()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    start = perf_counter()
    PREDICTIONS.inc()
    try:
        img_bgr = read_image_from_bytes(file.file.read())
        dets = detector.infer(img_bgr)  # [x1,y1,x2,y2,conf,cls_id]
        tracks = tracker.update(dets)   # adds track_id

        # prefer tracked boxes when available else raw detections
        by_id = {(t[6] if len(t)>=7 else None): t for t in tracks}
        output = []
        used = set()
        # include tracks
        for _, t in by_id.items():
            x1,y1,x2,y2,conf,cls_id,tid = t
            output.append(BBox(x1=x1,y1=y1,x2=x2,y2=y2,confidence=conf,cls=detector.cls_to_name(cls_id),track_id=tid))
            used.add(tuple(t[:5]))
        # include any untracked detections
        for d in dets:
            key = tuple(d[:5])
            if key in used: 
                continue
            x1,y1,x2,y2,conf,cls_id = d
            output.append(BBox(x1=x1,y1=y1,x2=x2,y2=y2,confidence=conf,cls=detector.cls_to_name(cls_id)))

        return PredictResponse(detections=output)
    except Exception as e:
        ERRORS.inc()
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        LATENCY.observe(perf_counter() - start)
        file.file.close()
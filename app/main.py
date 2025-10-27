from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from loguru import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from time import perf_counter
import numpy as np
import os
from .schemas import PredictResponse, BBox
from .inference import Detector
from .tracker import Tracker
from .utils import read_image_from_bytes
from .drift_detection import DriftDetector
from .nlp_query import NLPQueryEngine
from .metrics import (
    PREDICTIONS, ERRORS, LATENCY,
    MAP_50_95, MAP_50, MAP_75, MAP_VP_50_95,
    DETECTIONS_BY_CLASS, VEHICLE_PEDESTRIAN_COUNT
)

app = FastAPI(title="Autonomous Vision AI", version="0.1.0")

detector = Detector(model_name="yolov8l.pt")  # yolov8l.pt for best accuracy/speed trade-off
tracker = Tracker()
evaluator = None  # Lazy-loaded when evaluation endpoint is called
drift_detector = DriftDetector(baseline_size=1000, window_size=100)
nlp_engine = NLPQueryEngine()

@app.on_event("startup")
def startup_event():
    logger.info("Starting up...")
    # warmup
    detector.infer(np.zeros((640,640,3), np.uint8))
    logger.info("Startup complete.")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/")
def root():
    return {"message":"Welcome to Autonomous Vision AI"}

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
        tracks = tracker.update(dets, frame=img_bgr)   # adds track_id with actual frame for embeddings

        # prefer tracked boxes when available else raw detections
        by_id = {(t[6] if len(t)>=7 else None): t for t in tracks}
        output = []
        used = set()
        # include tracks
        for _, t in by_id.items():
            x1,y1,x2,y2,conf,cls_id,tid = t
            cls_name = detector.cls_to_name(cls_id)
            output.append(BBox(x1=x1,y1=y1,x2=x2,y2=y2,confidence=conf,cls=cls_name,track_id=tid))
            used.add(tuple(t[:5]))

            # Track metrics
            DETECTIONS_BY_CLASS.labels(class_name=cls_name).inc()
            if cls_id in detector.vehicle_pedestrian_classes:
                VEHICLE_PEDESTRIAN_COUNT.inc()

        # include any untracked detections
        for d in dets:
            key = tuple(d[:5])
            if key in used:
                continue
            x1,y1,x2,y2,conf,cls_id = d
            cls_name = detector.cls_to_name(cls_id)
            output.append(BBox(x1=x1,y1=y1,x2=x2,y2=y2,confidence=conf,cls=cls_name))

            # Track metrics
            DETECTIONS_BY_CLASS.labels(class_name=cls_name).inc()
            if cls_id in detector.vehicle_pedestrian_classes:
                VEHICLE_PEDESTRIAN_COUNT.inc()

        # Track drift
        confidences = [d.confidence for d in output]
        classes = [detector.class_names.get(d.cls, -1) for d in output if d.cls in [detector.class_names.get(k) for k in detector.class_names]]
        if confidences and classes:
            # Convert class names back to IDs for drift detection
            class_ids = []
            for d in output:
                for cid, cname in detector.class_names.items():
                    if cname == d.cls:
                        class_ids.append(cid)
                        break
            drift_detector.add_prediction(confidences, class_ids)

        # Add to NLP query history
        detection_dicts = [{"cls": d.cls, "confidence": d.confidence, "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2} for d in output]
        nlp_engine.add_detection_result(detection_dicts)

        return PredictResponse(detections=output)
    except Exception as e:
        ERRORS.inc()
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        LATENCY.observe(perf_counter() - start)
        file.file.close()


@app.post("/evaluate")
def evaluate_coco(
    coco_val_path: str = "/path/to/coco/val2017",
    coco_annotations_path: str = "/path/to/coco/annotations/instances_val2017.json",
    max_images: int = None,
    vehicle_pedestrian_only: bool = True
):
    """
    Run COCO evaluation on validation set

    Args:
        coco_val_path: Path to COCO val2017 images directory
        coco_annotations_path: Path to instances_val2017.json
        max_images: Limit evaluation to N images (for testing)
        vehicle_pedestrian_only: Only evaluate vehicle/pedestrian classes

    Returns:
        mAP metrics dictionary
    """
    global evaluator

    try:
        # Create evaluator with provided paths (recreate each time to allow different paths)
        from .evaluator import COCOEvaluator
        evaluator = COCOEvaluator(detector, coco_val_path, coco_annotations_path)

        # Run evaluation
        if vehicle_pedestrian_only:
            metrics = evaluator.evaluate_vehicle_pedestrian(max_images=max_images)
            # Update Prometheus metrics
            if "mAP_0.5_0.95_vp" in metrics:
                MAP_VP_50_95.set(metrics["mAP_0.5_0.95_vp"])
        else:
            metrics = evaluator.evaluate(max_images=max_images)
            # Update Prometheus metrics
            if "mAP_0.5_0.95" in metrics:
                MAP_50_95.set(metrics["mAP_0.5_0.95"])
            if "mAP_0.5" in metrics:
                MAP_50.set(metrics["mAP_0.5"])
            if "mAP_0.75" in metrics:
                MAP_75.set(metrics["mAP_0.75"])

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/drift/check")
def check_drift():
    """
    Run drift detection and return current status

    Returns:
        Drift detection results including KS test and PSI metrics
    """
    result = drift_detector.check_drift()
    return result


@app.get("/drift/status")
def drift_status():
    """
    Get current drift detector status

    Returns:
        Status information about drift detector
    """
    return drift_detector.get_status()


@app.get("/drift/history")
def drift_history(limit: int = 100):
    """
    Get drift detection history

    Args:
        limit: Maximum number of historical records to return

    Returns:
        List of historical drift detection results
    """
    return {"history": drift_detector.get_drift_history(limit)}


@app.post("/drift/reset")
def reset_drift_baseline():
    """
    Reset drift detector baseline

    This will clear the current baseline and start collecting a new one
    """
    drift_detector.reset_baseline()
    return {"status": "baseline_reset", "message": "Drift detector baseline has been reset"}


@app.post("/query")
def natural_language_query(question: str):
    """
    Query detection results and metrics using natural language

    Args:
        question: Natural language question about detections or metrics

    Returns:
        Answer to the question with relevant data

    Example queries:
    - "How many cars were detected in the last hour?"
    - "What's the class distribution of recent detections?"
    - "Show me high confidence detections"
    - "What's the current mAP score?"
    """
    try:
        result = nlp_engine.query(question)
        return result
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/query/stats")
def query_stats():
    """
    Get statistics about the NLP query engine

    Returns:
        Statistics about available data and query templates
    """
    return nlp_engine.get_stats()
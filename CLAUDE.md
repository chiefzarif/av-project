# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Autonomous Vision AI** FastAPI service that provides object detection and tracking capabilities using YOLOv8 and DeepSORT. The service accepts image uploads and returns detected objects with bounding boxes, confidence scores, and optional tracking IDs.

## Architecture

The codebase follows a modular structure:

- **`app/main.py`**: FastAPI application with three endpoints (`/health`, `/metrics`, `/predict`)
- **`app/inference.py`**: YOLOv8 object detection using Ultralytics
- **`app/tracker.py`**: Multi-object tracking using DeepSORT
- **`app/schemas.py`**: Pydantic models for API request/response
- **`app/metrics.py`**: Prometheus metrics for monitoring
- **`app/utils.py`**: Image processing utilities

### Core Flow
1. Image uploaded to `/predict` endpoint
2. `Detector.infer()` runs YOLOv8 detection → `[x1,y1,x2,y2,conf,cls_id]`
3. `Tracker.update()` adds tracking IDs → `[x1,y1,x2,y2,conf,cls_id,track_id]`
4. Response combines tracked and untracked detections as `BBox` objects

## Development Commands

### Running the Service
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload

# Run production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_health.py

# Run with verbose output
python -m pytest -v
```

### Metrics and Monitoring
- Prometheus metrics available at `/metrics` endpoint
- Tracks: prediction requests, errors, and latency
- Health check at `/health` endpoint

## Key Dependencies
- **FastAPI**: Web framework
- **Ultralytics**: YOLOv8 object detection
- **deep-sort-realtime**: Multi-object tracking
- **OpenCV**: Image processing
- **Prometheus Client**: Metrics collection
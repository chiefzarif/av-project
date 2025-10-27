# VisionGuard MLOps - Complete Documentation

Comprehensive documentation for the VisionGuard autonomous vision AI platform.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Docker Deployment](#docker-deployment)
3. [API Reference](#api-reference)
4. [Configuration](#configuration)
5. [Monitoring & Metrics](#monitoring--metrics)
6. [Cloud Deployment](#cloud-deployment)
7. [Development Guide](#development-guide)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- Docker (optional, recommended for production)
- 4GB+ RAM
- GPU (optional, for faster inference)

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd av-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --reload

# Service available at http://localhost:8000
```

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@zidane.jpg"
```

---

## Docker Deployment

### Basic Docker Usage

```bash
# Build image
docker build -t visionguard:latest .

# Run container
docker run -p 8000:8000 visionguard:latest

# Run with volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/coco:/app/coco:ro \
  -v $(pwd)/annotations:/app/annotations:ro \
  visionguard:latest
```

### Docker Compose (Recommended)

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f visionguard

# Stop service
docker-compose down

# Restart service
docker-compose restart
```

### With Monitoring Stack

```bash
# Start with Prometheus + Grafana
docker-compose --profile monitoring up -d

# Access services:
# - API: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### GPU Support

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

2. Uncomment GPU section in `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Run:
```bash
docker-compose up -d
```

---

## API Reference

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive API documentation (Swagger UI) |
| GET | `/health` | Health check |
| POST | `/predict` | Object detection & tracking |
| POST | `/evaluate` | COCO evaluation |
| GET | `/metrics` | Prometheus metrics |

### POST `/predict`

Perform object detection and tracking on an uploaded image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "x1": 121.34,
      "y1": 197.52,
      "x2": 1122.74,
      "y2": 711.43,
      "confidence": 0.9417,
      "cls": "person",
      "track_id": 5
    },
    {
      "x1": 746.12,
      "y1": 41.07,
      "x2": 1138.61,
      "y2": 710.59,
      "confidence": 0.9138,
      "cls": "person",
      "track_id": 3
    }
  ],
  "source_type": "image"
}
```

**Fields:**
- `x1, y1, x2, y2`: Bounding box coordinates (top-left and bottom-right)
- `confidence`: Detection confidence score (0-1)
- `cls`: Object class name (e.g., "person", "car", "truck")
- `track_id`: Persistent tracking ID (null if tracking failed)

**Status Codes:**
- `200 OK`: Successful detection
- `400 Bad Request`: Invalid image data
- `500 Internal Server Error`: Processing error

### POST `/evaluate`

Run COCO evaluation on validation dataset.

**Request:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "coco_val_path": "./coco/val2017",
    "coco_annotations_path": "./annotations/annotations/instances_val2017.json",
    "max_images": 5000,
    "vehicle_pedestrian_only": true
  }'
```

**Query Parameters:**
- `coco_val_path` (string): Path to COCO validation images
- `coco_annotations_path` (string): Path to COCO annotations JSON
- `max_images` (int, optional): Limit number of images to evaluate
- `vehicle_pedestrian_only` (bool): If true, only evaluate V/P classes

**Response:**
```json
{
  "status": "success",
  "metrics": {
    "mAP_0.5_0.95_vp": 0.5261,
    "mAP_0.5_vp": 0.6848,
    "mAP_0.75_vp": 0.5712,
    "mAP_small_vp": 0.4123,
    "mAP_medium_vp": 0.5891,
    "mAP_large_vp": 0.6534
  }
}
```

**Metrics Explained:**
- `mAP_0.5_0.95`: Mean Average Precision at IoU thresholds 0.5 to 0.95
- `mAP_0.5`: mAP at IoU threshold 0.5 (more lenient)
- `mAP_0.75`: mAP at IoU threshold 0.75 (stricter)
- `mAP_small/medium/large`: mAP by object size

### GET `/health`

Check service health status.

**Response:**
```json
{
  "status": "ok"
}
```

### GET `/metrics`

Prometheus metrics endpoint.

**Response:** (Plain text Prometheus format)
```
# HELP model_predictions_total Total prediction requests
# TYPE model_predictions_total counter
model_predictions_total 1523.0

# HELP model_map_50_95 mAP @ IoU=0.50:0.95 on COCO validation
# TYPE model_map_50_95 gauge
model_map_50_95 0.4936
...
```

---

## Configuration

### Model Selection

Edit `app/main.py`:

```python
# Available models:
# - yolov8n.pt: Nano (6MB, fastest, ~37% mAP)
# - yolov8s.pt: Small (22MB, fast, ~44% mAP)
# - yolov8m.pt: Medium (50MB, balanced, ~49% mAP) ⭐
# - yolov8l.pt: Large (84MB, accurate, ~52% mAP)
# - yolov8x.pt: Extra Large (131MB, most accurate, ~54% mAP)

detector = Detector(model_name="yolov8m.pt")
```

### Confidence Threshold

Edit `app/inference.py`:

```python
class Detector:
    def __init__(self, model_name: str = "yolov8m.pt", conf: float = 0.25):
        # conf=0.001: Maximum recall (evaluation)
        # conf=0.25: Balanced (recommended for production)
        # conf=0.4: High precision (fewer false positives)
```

### Vehicle/Pedestrian Classes

The system filters for these classes when `filter_classes=True`:

```python
vehicle_pedestrian_classes = {
    0: 'person',      # Pedestrians
    1: 'bicycle',     # Bicycles
    2: 'car',         # Cars
    3: 'motorcycle',  # Motorcycles
    5: 'bus',         # Buses
    7: 'truck'        # Trucks
}
```

**COCO Category Mapping:**
- YOLO class 0 → COCO category 1 (person)
- YOLO class 2 → COCO category 3 (car)
- YOLO class 5 → COCO category 6 (bus)
- etc.

### DeepSORT Tracking Parameters

Edit `app/tracker.py`:

```python
tracker = DeepSort(
    max_age=30,           # Frames to keep lost tracks
    n_init=2,             # Frames to confirm track
    nms_max_overlap=0.7,  # IoU threshold for NMS
    max_iou_distance=0.7  # Max IoU distance for matching
)
```

---

## Monitoring & Metrics

### Prometheus Metrics

Access at `http://localhost:8000/metrics`

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `model_predictions_total` | Counter | Total prediction requests |
| `model_errors_total` | Counter | Total errors |
| `model_latency_seconds` | Histogram | Request latency distribution |
| `model_map_50_95` | Gauge | Overall mAP @ IoU 0.5:0.95 |
| `model_map_50` | Gauge | mAP @ IoU 0.5 |
| `model_map_75` | Gauge | mAP @ IoU 0.75 |
| `model_map_vp_50_95` | Gauge | Vehicle/Pedestrian mAP |
| `model_detections_by_class` | Counter | Detections per class |
| `model_vehicle_pedestrian_detections_total` | Counter | Total V/P detections |

### Grafana Dashboards

1. **Start monitoring stack:**
```bash
docker-compose --profile monitoring up -d
```

2. **Access Grafana:** http://localhost:3000
   - Username: `admin`
   - Password: `admin`

3. **Add Prometheus data source:**
   - Configuration → Data Sources → Add data source
   - Select "Prometheus"
   - URL: `http://prometheus:9090`
   - Click "Save & Test"

4. **Create Dashboard:**

**Example Panel Queries:**

```promql
# Request rate (per second)
rate(model_predictions_total[5m])

# Error rate
rate(model_errors_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(model_latency_seconds_bucket[5m]))

# Current mAP
model_map_vp_50_95

# Detections by class
sum by (class_name) (model_detections_by_class)
```

---

## Cloud Deployment

### AWS EC2 Deployment

#### 1. Launch GPU Instance

```bash
# Create instance
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-0123456789abcdef \
  --subnet-id subnet-0123456789abcdef \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=VisionGuard}]'
```

#### 2. Install Docker

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<instance-ip>

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# For GPU: Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-docker2
sudo systemctl restart docker
```

#### 3. Deploy Application

```bash
# Clone repository
git clone <your-repo-url>
cd av-project

# Copy model weights (if not in repo)
scp -i your-key.pem yolov8m.pt ec2-user@<instance-ip>:~/av-project/

# Start service
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

#### 4. Configure Security Group

Allow inbound traffic on port 8000:
- Type: Custom TCP
- Port: 8000
- Source: Your IP or 0.0.0.0/0 (for public access)

### Google Cloud Platform Deployment

#### 1. Create VM with GPU

```bash
# Create instance
gcloud compute instances create visionguard-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

# SSH into VM
gcloud compute ssh visionguard-vm --zone=us-central1-a
```

#### 2. Install Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 3. Deploy Application

```bash
# Clone and deploy
git clone <your-repo-url>
cd av-project
docker-compose up -d
```

#### 4. Configure Firewall

```bash
# Allow HTTP traffic on port 8000
gcloud compute firewall-rules create allow-visionguard \
  --allow=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --description="Allow VisionGuard API traffic"
```

### Container Registry

#### Push to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag visionguard:latest your-username/visionguard:v1.0

# Push
docker push your-username/visionguard:v1.0
```

#### Push to AWS ECR

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag visionguard:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/visionguard:v1.0

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/visionguard:v1.0
```

#### Push to GCP Artifact Registry

```bash
# Configure Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Tag
docker tag visionguard:latest us-central1-docker.pkg.dev/<project-id>/visionguard/visionguard:v1.0

# Push
docker push us-central1-docker.pkg.dev/<project-id>/visionguard/visionguard:v1.0
```

---

## Development Guide

### Project Structure

```
av-project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints
│   ├── inference.py     # YOLOv8 detection
│   ├── tracker.py       # DeepSORT tracking
│   ├── evaluator.py     # COCO evaluation
│   ├── metrics.py       # Prometheus metrics
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # Utilities
├── tests/               # Unit tests
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Multi-container orchestration
├── requirements.txt     # Python dependencies
├── prometheus.yml       # Prometheus config
├── README.md            # Quick start guide
└── DOCUMENTATION.md     # This file
```

### Adding New Features

#### 1. Add New Endpoint

Edit `app/main.py`:

```python
@app.post("/new-endpoint")
async def new_endpoint(param: str):
    # Your logic here
    return {"result": "success"}
```

#### 2. Add New Model

```python
# Download model
from ultralytics import YOLO
model = YOLO("yolov8x.pt")  # Downloads automatically

# Update main.py
detector = Detector(model_name="yolov8x.pt")
```

#### 3. Add New Metric

Edit `app/metrics.py`:

```python
from prometheus_client import Counter

NEW_METRIC = Counter(
    'my_new_metric',
    'Description of metric'
)

# Use in code
NEW_METRIC.inc()
```

### Code Style

Follow PEP 8 guidelines:

```bash
# Install linters
pip install black flake8 mypy

# Format code
black app/

# Check style
flake8 app/

# Type checking
mypy app/
```

---

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_inference.py -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=html
```

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"

# Evaluation (requires COCO dataset)
curl -X POST "http://localhost:8000/evaluate? \
  coco_val_path=./coco/val2017& \
  coco_annotations_path=./annotations/annotations/instances_val2017.json& \
  max_images=100& \
  vehicle_pedestrian_only=true"
```

### Load Testing

#### Using wrk

```bash
# Install wrk
brew install wrk  # macOS
# or build from source

# Test endpoint
wrk -t4 -c100 -d30s http://localhost:8000/health

# With POST request
wrk -t4 -c100 -d30s -s post_script.lua http://localhost:8000/predict
```

#### Using Apache Bench

```bash
# Simple load test
ab -n 1000 -c 10 http://localhost:8000/health

# POST with file
ab -n 100 -c 10 -p test_image.jpg -T "image/jpeg" http://localhost:8000/predict
```

### Performance Benchmarking

```python
# Create benchmark script
import time
import requests

def benchmark_inference(num_requests=100):
    url = "http://localhost:8000/predict"
    times = []

    for i in range(num_requests):
        start = time.time()
        files = {'file': open('test_image.jpg', 'rb')}
        response = requests.post(url, files=files)
        times.append(time.time() - start)

    print(f"Average: {sum(times)/len(times):.3f}s")
    print(f"Min: {min(times):.3f}s")
    print(f"Max: {max(times):.3f}s")

benchmark_inference()
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Daemon Not Running

**Error:** `Cannot connect to the Docker daemon`

**Solution:**
```bash
# Start Docker
sudo systemctl start docker  # Linux
open -a Docker              # macOS
```

#### 2. Port Already in Use

**Error:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
docker run -p 8001:8000 visionguard:latest
```

#### 3. Out of Memory

**Error:** `Killed` or container exits with code 137

**Solution:**
```bash
# Increase Docker memory limit (Docker Desktop)
# Or use smaller model
detector = Detector(model_name="yolov8n.pt")
```

#### 4. Model Download Fails

**Error:** Connection timeout downloading model

**Solution:**
```bash
# Manually download
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Copy to Docker image
COPY yolov8m.pt /app/yolov8m.pt
```

#### 5. CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
```python
# Reduce batch size or use CPU
import torch
torch.cuda.empty_cache()

# Or force CPU
device = 'cpu'
```

### Logging

#### Enable Debug Logging

```python
# In app/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### View Container Logs

```bash
# Docker Compose
docker-compose logs -f visionguard

# Docker
docker logs -f <container-id>

# Last 100 lines
docker logs --tail 100 visionguard-api
```

---

## Performance Optimization

### Model Optimization

#### 1. Use Smaller Models

```python
# Fastest (6MB)
detector = Detector(model_name="yolov8n.pt")

# Balanced (50MB)
detector = Detector(model_name="yolov8m.pt")
```

#### 2. Adjust Confidence Threshold

```python
# Higher threshold = faster (fewer detections)
detector = Detector(conf=0.5)
```

#### 3. Disable Tracking

```python
# In app/main.py
# Comment out tracker.update() if tracking not needed
detections = detector.infer(img_bgr)
# tracks = tracker.update(dets, frame=img_bgr)  # Commented out
```

### Infrastructure Optimization

#### 1. GPU Acceleration

Provides 5-10x speedup:

```yaml
# docker-compose.yml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

#### 2. Batch Processing

For multiple images:

```python
# Future enhancement
results = model.predict(images, batch=16)
```

#### 3. Model Quantization

Reduce model size and increase speed:

```python
# Export to TensorRT (future enhancement)
model.export(format='engine', half=True)
```

### Caching

#### Redis Cache (future enhancement)

```python
import redis
cache = redis.Redis(host='localhost', port=6379)

# Cache results
cache.set(f"detection:{image_hash}", json.dumps(detections))
```

---

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [COCO Dataset](https://cocodataset.org/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Docker Documentation](https://docs.docker.com/)

---

**Last Updated:** October 21, 2025
**Version:** 1.0.0
**Maintainer:** VisionGuard Team

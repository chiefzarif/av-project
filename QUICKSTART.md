# Quick Start Guide

Get the Autonomous Vision AI system running in 5 minutes.

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Kubernetes cluster (for K8s deployment)

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Service

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Test the API

```bash
# Health check
curl http://127.0.0.1:8000/health

# Run object detection
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@zidane.jpg"
```

### 4. Run Full Demo

```bash
chmod +x demo.sh
./demo.sh
```

## Docker Deployment

### Build Image

```bash
# CPU-only
docker build -t autonomous-vision-ai:latest .

# GPU-enabled (requires nvidia-docker)
docker build -t autonomous-vision-ai:gpu .
```

### Run Container

```bash
# CPU
docker run -p 8000:8000 autonomous-vision-ai:latest

# GPU
docker run --gpus all -p 8000:8000 autonomous-vision-ai:gpu
```

### Docker Compose

```bash
docker-compose up
```

This starts:
- FastAPI service (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)

## Kubernetes Deployment

### 1. Apply Configurations

```bash
# Apply all K8s resources
kubectl apply -f k8s/

# Or individually
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 2. Verify Deployment

```bash
# Check pods
kubectl get pods -l app=autonomous-vision

# Check service
kubectl get svc autonomous-vision-service

# Check HPA
kubectl get hpa
```

### 3. Access Service

```bash
# Port forward for local access
kubectl port-forward svc/autonomous-vision-service 8000:80

# Or use LoadBalancer/Ingress (see k8s/ingress.yaml)
```

## Key Features

### 1. Object Detection & Tracking

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test_image.jpg"
```

### 2. Drift Detection

```bash
# Check for drift
curl http://127.0.0.1:8000/drift/check

# View drift status
curl http://127.0.0.1:8000/drift/status

# View drift history
curl http://127.0.0.1:8000/drift/history
```

### 3. Natural Language Queries

```bash
# Ask questions about detections
curl -X POST "http://127.0.0.1:8000/query?question=How%20many%20cars%20detected?"

# Example queries
curl -X POST "http://127.0.0.1:8000/query?question=What's%20the%20class%20distribution?"
curl -X POST "http://127.0.0.1:8000/query?question=Show%20me%20recent%20detections"
```

### 4. Model Evaluation

```bash
# Run COCO evaluation (requires COCO dataset)
curl -X POST "http://127.0.0.1:8000/evaluate?coco_val_path=./coco/val2017&coco_annotations_path=./annotations/instances_val2017.json&max_images=500&vehicle_pedestrian_only=true"
```

### 5. Prometheus Metrics

```bash
# View all metrics
curl http://127.0.0.1:8000/metrics

# Prometheus UI (if using docker-compose)
open http://localhost:9090
```

### 6. Grafana Dashboard

```bash
# Grafana UI (if using docker-compose)
open http://localhost:3000

# Import dashboard: grafana-dashboard.json
```

## Performance Testing

### Load Test

```bash
cd k8s
./load-test.sh
```

This will:
- Generate traffic to test autoscaling
- Monitor HPA scaling behavior
- Verify system performance under load

## Monitoring

### View Logs

```bash
# Local
tail -f logs/app.log

# Docker
docker logs -f <container_id>

# Kubernetes
kubectl logs -f deployment/autonomous-vision
```

### Metrics

Access Prometheus metrics at `/metrics` endpoint:

- `predictions_total`: Total prediction requests
- `prediction_errors_total`: Total errors
- `prediction_latency`: Request latency histogram
- `map_vehicle_pedestrian_50_95`: mAP score for vehicle/pedestrian classes
- `detections_by_class_total`: Detections grouped by class
- `vehicle_pedestrian_detections_total`: Total V/P detections

## Troubleshooting

### Service won't start

```bash
# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip install -r requirements.txt

# Check port availability
lsof -i :8000
```

### GPU not detected

```bash
# Check CUDA
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Kubernetes pods not starting

```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Check resources
kubectl top nodes
kubectl top pods
```

## Next Steps

1. **Set up COCO dataset** for full evaluation
2. **Configure Grafana alerts** for drift detection
3. **Tune HPA settings** for your workload
4. **Set up CI/CD pipeline** for automated deployments
5. **Implement A/B testing** for model updates

## Documentation

- [README.md](README.md) - Project overview
- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) - Detailed documentation
- [k8s/README.md](k8s/README.md) - Kubernetes deployment guide
- [docs/ROADMAP.md](docs/ROADMAP.md) - Future enhancements

## Support

For issues or questions:
1. Check existing documentation
2. Review GitHub issues
3. Check application logs

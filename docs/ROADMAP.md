# VisionGuard MLOps Roadmap

This document outlines the step-by-step roadmap to build the complete VisionGuard MLOps platform.

## Phase 1: Core CV Model + Serving ✅ (WEEKS 1-2)

**Goal**: Get a working YOLOv8 + DeepSORT pipeline served through FastAPI.

### Completed ✅
- [x] YOLOv8 object detection implemented ([`app/inference.py`](app/inference.py))
- [x] DeepSORT multi-object tracking ([`app/tracker.py`](app/tracker.py))
- [x] FastAPI microservice with endpoints ([`app/main.py`](app/main.py))
- [x] COCO evaluation system ([`app/evaluator.py`](app/evaluator.py))
- [x] Prometheus metrics ([`app/metrics.py`](app/metrics.py))
- [x] Basic testing framework ([`tests/`](tests/))

### Next Steps to Complete Phase 1
1. **Run COCO Evaluation** 🎯
   ```bash
   # Test current model performance
   curl -X POST "http://localhost:8000/evaluate" \
        -H "Content-Type: application/json" \
        -d '{
          "coco_val_path": "./coco/val2017",
          "coco_annotations_path": "./coco/annotations/instances_val2017.json",
          "max_images": 500,
          "vehicle_pedestrian_only": true
        }'
   ```

2. **Optimize for 90%+ mAP**
   - Benchmark YOLOv8m vs YOLOv8l models
   - Fine-tune confidence thresholds
   - Add model ensemble if needed

3. **Performance Testing**
   ```bash
   # Test latency requirements (<200ms)
   curl -X POST "http://localhost:8000/predict" \
        -F "file=@test_image.jpg" \
        --header "Content-Type: multipart/form-data"
   ```

**Resume Win**: "Built real-time detection + tracking API with YOLOv8 + DeepSORT (90% mAP, <200ms latency)."

---

## Phase 2: Containerization + Basic Deployment (WEEK 3)

**Goal**: Make the service portable & deployable.

### Steps

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   # Install system dependencies for OpenCV
   RUN apt-get update && apt-get install -y \
       libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
       libglib2.0-0 libgtk-3-0 libgdk-pixbuf2.0-0 \
       && rm -rf /var/lib/apt/lists/*

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .
   
   EXPOSE 8000
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and Test Locally**
   ```bash
   # Build image
   docker build -t visionguard:latest .
   
   # Test locally
   docker run -p 8000:8000 visionguard:latest
   
   # Test with GPU (if available)
   docker run --gpus all -p 8000:8000 visionguard:latest
   ```

3. **Push to Registry**
   ```bash
   # Tag and push
   docker tag visionguard:latest your-registry/visionguard:v1.0
   docker push your-registry/visionguard:v1.0
   ```

4. **Cloud Deployment**
   ```bash
   # AWS EC2 with GPU
   aws ec2 run-instances --image-id ami-xxx --instance-type g4dn.xlarge
   
   # Or GCP with GPU
   gcloud compute instances create visionguard-vm \
     --accelerator type=nvidia-tesla-t4,count=1 \
     --machine-type n1-standard-4
   ```

**Resume Win**: Adds Docker + cloud deployment keywords.

---

## Phase 3: Kubernetes + Scaling (WEEKS 4-5)

**Goal**: Production-ready orchestration with autoscaling.

### Steps

1. **Set Up Kubernetes Cluster**
   ```bash
   # Option 1: Local with Kind
   kind create cluster --config kind-gpu-config.yaml
   
   # Option 2: EKS
   eksctl create cluster --name visionguard --nodes-min 1 --nodes-max 5 --node-type g4dn.xlarge
   
   # Option 3: GKE
   gcloud container clusters create visionguard \
     --accelerator type=nvidia-tesla-t4,count=1 \
     --machine-type n1-standard-4
   ```

2. **Create Kubernetes Manifests**
   ```yaml
   # k8s/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: visionguard
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: visionguard
     template:
       metadata:
         labels:
           app: visionguard
       spec:
         containers:
         - name: visionguard
           image: your-registry/visionguard:v1.0
           ports:
           - containerPort: 8000
           resources:
             requests:
               nvidia.com/gpu: 1
               memory: "4Gi"
               cpu: "1000m"
             limits:
               nvidia.com/gpu: 1
               memory: "8Gi"
               cpu: "2000m"
   ```

3. **Configure HPA**
   ```yaml
   # k8s/hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: visionguard-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: visionguard
     minReplicas: 1
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

4. **Load Testing**
   ```bash
   # Install wrk
   brew install wrk
   
   # Run load test
   wrk -t12 -c400 -d30s --script=load_test.lua http://visionguard-service:8000/predict
   ```

**Resume Win**: "Deployed GPU-accelerated inference service on Kubernetes with HPA scaling to 1,000+ FPS throughput."

---

## Phase 4: Monitoring + Drift Detection (WEEK 6)

**Goal**: Add observability + ML monitoring.

### Steps

1. **Enhanced Prometheus Metrics**
   ```python
   # Add to app/metrics.py
   from prometheus_client import Histogram, Counter
   
   # Drift detection metrics
   DRIFT_DETECTED = Counter('model_drift_detected_total', 'Number of drift detections')
   INPUT_DISTRIBUTION = Histogram('input_distribution_ks_statistic', 'KS test statistic for input drift')
   PREDICTION_CONFIDENCE = Histogram('prediction_confidence_distribution', 'Distribution of prediction confidences')
   ```

2. **Implement Drift Detection**
   ```python
   # Create app/drift_detection.py
   import numpy as np
   from scipy import stats
   
   class DriftDetector:
       def __init__(self, reference_data):
           self.reference_data = reference_data
           
       def detect_drift(self, new_data, threshold=0.05):
           """Use Kolmogorov-Smirnov test for drift detection"""
           ks_statistic, p_value = stats.ks_2samp(self.reference_data, new_data)
           return p_value < threshold, ks_statistic
   ```

3. **Grafana Dashboard Setup**
   ```bash
   # Deploy Prometheus and Grafana
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm install prometheus prometheus-community/kube-prometheus-stack
   
   # Access Grafana
   kubectl port-forward svc/prometheus-grafana 3000:80
   ```

4. **Create Grafana Dashboards**
   - Model performance metrics (mAP, latency)
   - Infrastructure metrics (GPU utilization, memory)
   - Drift detection alerts
   - Request throughput and error rates

**Resume Win**: "Implemented model monitoring with drift detection, Prometheus metrics, Grafana dashboards."

---

## Phase 5: LLM Log Intelligence (WEEKS 7-8)

**Goal**: Human-friendly log analysis with LLM-powered insights.

### Steps

1. **Set Up Log Collection**
   ```python
   # Add structured logging to app/main.py
   import structlog
   
   logger = structlog.get_logger()
   
   @app.middleware("http")
   async def log_requests(request: Request, call_next):
       start_time = time.time()
       response = await call_next(request)
       process_time = time.time() - start_time
       
       logger.info("request_processed",
                  method=request.method,
                  url=str(request.url),
                  status_code=response.status_code,
                  process_time=process_time)
       return response
   ```

2. **Create LLM Log Intelligence Service**
   ```python
   # Create llm_service/main.py
   from langchain.llms import OpenAI
   from langchain.chains import LLMChain
   from langchain.prompts import PromptTemplate
   
   class LogIntelligence:
       def __init__(self):
           self.llm = OpenAI(temperature=0.7)
           
       def analyze_anomalies(self, log_data):
           prompt = PromptTemplate(
               input_variables=["logs"],
               template="""
               Analyze these system logs and identify anomalies:
               {logs}
               
               Provide:
               1. Summary of issues found
               2. Potential root causes
               3. Recommended actions
               """
           )
           chain = LLMChain(llm=self.llm, prompt=prompt)
           return chain.run(logs=log_data)
   ```

3. **Add RAG for Historical Analysis**
   ```python
   # Add to llm_service/rag.py
   from langchain.vectorstores import FAISS
   from langchain.embeddings import OpenAIEmbeddings
   
   class LogRAG:
       def __init__(self):
           self.embeddings = OpenAIEmbeddings()
           self.vectorstore = FAISS(embedding_function=self.embeddings)
           
       def index_logs(self, historical_logs):
           """Index historical logs for similarity search"""
           self.vectorstore.add_texts(historical_logs)
           
       def query_similar_incidents(self, current_issue):
           """Find similar historical incidents"""
           docs = self.vectorstore.similarity_search(current_issue, k=5)
           return docs
   ```

4. **Create Report Endpoint**
   ```python
   # Add to app/main.py
   @app.post("/report")
   async def generate_report(time_range: str = "24h"):
       logs = collect_logs(time_range)
       intelligence = LogIntelligence()
       report = intelligence.analyze_anomalies(logs)
       return {"report": report, "timestamp": datetime.now()}
   ```

**Resume Win**: "Built LLM-powered log intelligence service, reducing debugging & retraining cycles by 50%."

---

## Testing and Validation

### Performance Benchmarks
- **mAP Target**: >90% on COCO vehicle/pedestrian classes
- **Latency Target**: <200ms inference time
- **Throughput Target**: 1,000+ FPS with autoscaling
- **Availability Target**: 99.9% uptime

### Load Testing Scripts
```bash
# Create load_test.lua for wrk
-- load_test.lua
wrk.method = "POST"
wrk.body = read_file("test_image.jpg")
wrk.headers["Content-Type"] = "image/jpeg"
```

### Integration Tests
```python
# tests/test_e2e.py
def test_full_pipeline():
    # Test image upload → detection → tracking → response
    response = client.post("/predict", files={"file": test_image})
    assert response.status_code == 200
    assert len(response.json()["detections"]) > 0
```

---

## Resume Achievement Mapping

✅ **"Achieved 90%+ mAP on COCO dataset"** → Phase 1 completion  
✅ **"GPU-accelerated FastAPI microservices on Kubernetes with HPA"** → Phase 3 completion  
✅ **"Prometheus/Grafana observability"** → Phase 4 completion  
✅ **"LLM-powered log intelligence system using LangChain and FAISS RAG"** → Phase 5 completion  
✅ **"Reduced log analysis time by 40%"** → Measured through Phase 5 implementation

---

## Current Status Tracking

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Core CV Model + Serving | ✅ Complete | 100% |
| Phase 2: Containerization + Deployment | ✅ Complete | 100% |
| Phase 3: Kubernetes + Scaling | ✅ Complete | 100% |
| Phase 4: Monitoring + Drift Detection | ⚪ Not Started | 0% |
| Phase 5: LLM Log Intelligence | ⚪ Not Started | 0% |

**Latest Achievement**: Phase 3 complete with Kubernetes orchestration, HPA autoscaling, and load testing framework.

**Next Priority**: Phase 4 - Enhanced monitoring with Prometheus/Grafana stack and drift detection.
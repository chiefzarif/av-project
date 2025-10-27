# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying VisionGuard MLOps on Kubernetes clusters.

## 📁 Files

| File | Description |
|------|-------------|
| `deployment.yaml` | Main deployment with CPU and GPU variants |
| `service.yaml` | ClusterIP, LoadBalancer, and headless services |
| `hpa.yaml` | HorizontalPodAutoscaler for autoscaling |
| `configmap.yaml` | Configuration for model and tracking parameters |
| `ingress.yaml` | Ingress with TLS and network policies |
| `load-test.sh` | Load testing script |
| `load-test.lua` | WRK Lua script for load testing |

## 🚀 Quick Start

### 1. Local Testing with Kind

```bash
# Create local cluster
kind create cluster --name visionguard

# Build and load image
docker build -t visionguard:latest ..
kind load docker-image visionguard:latest --name visionguard

# Deploy
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Check deployment
kubectl get pods -w
kubectl get svc
```

### 2. Cloud Deployment (GKE)

```bash
# Create GKE cluster
gcloud container clusters create visionguard \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10 \
  --region us-central1

# Push image to GCR
docker tag visionguard:latest gcr.io/PROJECT_ID/visionguard:latest
docker push gcr.io/PROJECT_ID/visionguard:latest

# Update deployment.yaml image reference
# Then apply manifests
kubectl apply -f .
```

### 3. Cloud Deployment (EKS)

```bash
# Create EKS cluster
eksctl create cluster \
  --name visionguard \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed

# Push image to ECR
aws ecr create-repository --repository-name visionguard
docker tag visionguard:latest ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/visionguard:latest
docker push ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/visionguard:latest

# Deploy
kubectl apply -f .
```

## 🎯 Deployment Variants

### Standard (CPU-only)
```bash
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
```

### GPU-accelerated
```bash
# Requires GPU nodes in cluster
kubectl label nodes <node-name> cloud.google.com/gke-accelerator=nvidia-tesla-t4

# Deploy GPU variant (included in deployment.yaml)
kubectl get deployment visionguard-gpu
```

### With Ingress
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Apply ingress
kubectl apply -f ingress.yaml

# Get external IP
kubectl get ingress visionguard-ingress
```

## 📊 Monitoring & Scaling

### Check HPA Status
```bash
kubectl get hpa visionguard-hpa
kubectl describe hpa visionguard-hpa
```

### View Autoscaling Events
```bash
kubectl get events --sort-by='.lastTimestamp' | grep HorizontalPodAutoscaler
```

### Manual Scaling
```bash
# Scale manually (overrides HPA temporarily)
kubectl scale deployment visionguard --replicas=5
```

### Resource Usage
```bash
kubectl top pods
kubectl top nodes
```

## 🧪 Load Testing

### Prerequisites
```bash
# Install wrk
brew install wrk  # macOS
sudo apt-get install wrk  # Ubuntu
```

### Run Load Test
```bash
# From project root
./k8s/load-test.sh

# With custom parameters
SERVICE_URL=http://visionguard-lb THREADS=8 CONNECTIONS=200 DURATION=60s ./k8s/load-test.sh

# Using wrk directly
wrk -t4 -c100 -d30s -s k8s/load-test.lua http://visionguard:8000/predict
```

### Expected Performance
- **Latency**: P95 < 200ms
- **Throughput**: > 50 RPS per pod
- **Error Rate**: < 1%

## 🔧 Configuration

### Update ConfigMap
```bash
# Edit configuration
kubectl edit configmap visionguard-config

# Restart pods to apply changes
kubectl rollout restart deployment visionguard
```

### Environment Variables
Configure in `deployment.yaml`:
- `MODEL_VARIANT`: yolov8n, yolov8s, yolov8m, yolov8l (default)
- `CONFIDENCE_THRESHOLD`: Detection confidence (default: 0.25)
- `PYTHONUNBUFFERED`: Enable real-time logging (default: 1)

## 🛡️ Security

### Network Policies
```bash
# Apply network policy
kubectl apply -f ingress.yaml  # Contains NetworkPolicy

# Verify
kubectl get networkpolicies
kubectl describe networkpolicy visionguard-netpol
```

### TLS/SSL
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create Let's Encrypt issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Update ingress.yaml annotations and apply
```

## 📈 Scaling Strategy

### HPA Configuration
```yaml
minReplicas: 2
maxReplicas: 10
metrics:
  - CPU: 70%
  - Memory: 80%
  - Custom: prediction_requests_per_second > 100
```

### Scale-Up Behavior
- **Aggressive**: +100% or +4 pods every 30s
- **Policy**: Fastest scale-up wins

### Scale-Down Behavior
- **Conservative**: -50% or -2 pods every 60s
- **Stabilization**: 5-minute window
- **Policy**: Slowest scale-down wins

## 🐛 Troubleshooting

### Pods Not Starting
```bash
# Check pod status
kubectl get pods -l app=visionguard
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

### Service Not Accessible
```bash
# Check service
kubectl get svc visionguard
kubectl describe svc visionguard

# Port forward for testing
kubectl port-forward svc/visionguard 8000:8000
curl http://localhost:8000/health
```

### HPA Not Scaling
```bash
# Check metrics server
kubectl top nodes
kubectl top pods

# Check HPA
kubectl describe hpa visionguard-hpa

# View HPA events
kubectl get events --field-selector involvedObject.name=visionguard-hpa
```

### Image Pull Errors
```bash
# Check image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email>

# Add to deployment.yaml:
# spec:
#   imagePullSecrets:
#   - name: regcred
```

## 🔄 Rolling Updates

### Update Deployment
```bash
# Update image
kubectl set image deployment/visionguard visionguard=visionguard:v2.0

# Or apply updated manifest
kubectl apply -f deployment.yaml

# Watch rollout
kubectl rollout status deployment/visionguard

# Check rollout history
kubectl rollout history deployment/visionguard
```

### Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/visionguard

# Rollback to specific revision
kubectl rollout undo deployment/visionguard --to-revision=2
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [HPA v2 Spec](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [NGINX Ingress](https://kubernetes.github.io/ingress-nginx/)
- [Cert-Manager](https://cert-manager.io/)
- [VisionGuard Documentation](../DOCUMENTATION.md)

## 💡 Best Practices

1. **Always use resource requests/limits** - Ensures proper scheduling and QoS
2. **Enable HPA** - Automatic scaling based on load
3. **Use readiness probes** - Prevent traffic to unhealthy pods
4. **Use liveness probes** - Restart unhealthy pods
5. **Configure PodDisruptionBudgets** - Ensure availability during cluster maintenance
6. **Use NetworkPolicies** - Restrict network access
7. **Enable monitoring** - Prometheus + Grafana for observability
8. **Version your images** - Never use `latest` in production
9. **Test rollouts in staging** - Validate before production deployment
10. **Monitor costs** - Use cluster autoscaler and right-size resources

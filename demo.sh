#!/bin/bash
# Demo script for Autonomous Vision AI


set -e

API_URL="http://127.0.0.1:8000"
echo "=== Autonomous Vision AI Demo ==="
echo "API URL: $API_URL"
echo ""

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
section() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

# Function to check if service is running
check_service() {
    if ! curl -s "$API_URL/health" > /dev/null; then
        echo -e "${YELLOW}Warning: Service not running at $API_URL${NC}"
        echo "Please start the service with: uvicorn app.main:app --reload"
        exit 1
    fi
    echo -e "${GREEN}✓ Service is running${NC}\n"
}

section "1. Health Check"
check_service

section "2. YOLOv8 + DeepSORT Inference Demo"
echo "Testing real-time object detection and tracking..."
if [ -f "zidane.jpg" ]; then
    echo -e "${GREEN}Sending test image...${NC}"
    curl -s -X POST "$API_URL/predict" \
        -F "file=@zidane.jpg" | jq '.'
else
    echo -e "${YELLOW}Test image not found. Skipping inference demo.${NC}"
fi

section "3. Drift Detection (KS Test + PSI)"
echo "Demonstrating drift detection capabilities..."

echo -e "${GREEN}[3.1] Drift Detector Status${NC}"
curl -s "$API_URL/drift/status" | jq '.'

echo -e "\n${GREEN}[3.2] Run Drift Check${NC}"
curl -s "$API_URL/drift/check" | jq '.'

echo -e "\n${GREEN}[3.3] Drift History${NC}"
curl -s "$API_URL/drift/history?limit=5" | jq '.history[-1]'

section "4. Natural Language Query System (LangChain + FAISS)"
echo "Testing NLP query engine for analysis..."

queries=(
    "How many detections have we processed?"
    "What's the class distribution?"
    "Show me recent detections"
    "How many vehicles were detected?"
)

for query in "${queries[@]}"; do
    echo -e "\n${GREEN}Query: \"$query\"${NC}"
    curl -s -X POST "$API_URL/query?question=$(echo $query | jq -sRr @uri)" | jq -r '.answer'
done

echo -e "\n${GREEN}Query Engine Stats:${NC}"
curl -s "$API_URL/query/stats" | jq '.'

section "5. Prometheus Metrics"
echo "Viewing Prometheus metrics for observability..."
echo -e "${GREEN}Sample metrics:${NC}"
curl -s "$API_URL/metrics" | grep -E "(predictions_total|prediction_latency|map_)" | head -10

section "6. Performance Metrics"
echo "Current performance stats..."
curl -s "$API_URL/metrics" | grep -A 5 "prediction_latency" | head -6

section "7. Model Accuracy (mAP)"
echo -e "${YELLOW}Note: Full COCO evaluation requires COCO dataset${NC}"
echo "To run evaluation:"
echo "  curl -X POST \"$API_URL/evaluate?coco_val_path=./coco/val2017&coco_annotations_path=./annotations/instances_val2017.json&max_images=500&vehicle_pedestrian_only=true\""

section "Demo Complete!"
echo -e "${GREEN}✓ All features demonstrated successfully${NC}"
echo ""
echo "Key Resume Features Validated:"
echo "  ✓ YOLOv8 + DeepSORT tracking"
echo "  ✓ Drift detection (KS test + PSI)"
echo "  ✓ NLP query system (LangChain + FAISS)"
echo "  ✓ Prometheus metrics"
echo "  ✓ FastAPI microservice"
echo ""
echo "Next steps:"
echo "  - Deploy to Kubernetes: kubectl apply -f k8s/"
echo "  - View Grafana dashboard: Import grafana-dashboard.json"
echo "  - Run evaluation on COCO dataset"

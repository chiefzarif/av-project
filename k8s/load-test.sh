#!/bin/bash
# Load testing script for VisionGuard API
# Requires: wrk (brew install wrk on macOS)

set -e

# Configuration
SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"
THREADS="${THREADS:-4}"
CONNECTIONS="${CONNECTIONS:-100}"
DURATION="${DURATION:-30s}"
TEST_IMAGE="${TEST_IMAGE:-zidane.jpg}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}VisionGuard Load Test${NC}"
echo -e "${BLUE}==================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Service URL:  $SERVICE_URL"
echo "  Threads:      $THREADS"
echo "  Connections:  $CONNECTIONS"
echo "  Duration:     $DURATION"
echo "  Test Image:   $TEST_IMAGE"
echo ""

# Check if wrk is installed
if ! command -v wrk &> /dev/null; then
    echo -e "${YELLOW}wrk not found. Installing...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install wrk
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y wrk
    else
        echo "Please install wrk manually"
        exit 1
    fi
fi

# Check if test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${YELLOW}Test image not found. Using default...${NC}"
    TEST_IMAGE="zidane.jpg"
fi

# Health check
echo -e "${GREEN}Performing health check...${NC}"
if curl -s -f "$SERVICE_URL/health" > /dev/null; then
    echo -e "${GREEN}✓ Service is healthy${NC}"
else
    echo -e "${YELLOW}✗ Service health check failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Starting load test...${NC}"
echo ""

# Run load test
if [ -f "k8s/load-test.lua" ]; then
    wrk -t"$THREADS" -c"$CONNECTIONS" -d"$DURATION" \
        -s k8s/load-test.lua \
        "$SERVICE_URL/predict"
else
    # Simple health endpoint test if Lua script not found
    echo -e "${YELLOW}Lua script not found. Testing health endpoint instead${NC}"
    wrk -t"$THREADS" -c"$CONNECTIONS" -d"$DURATION" \
        "$SERVICE_URL/health"
fi

echo ""
echo -e "${GREEN}Load test complete!${NC}"
echo ""

# Get final metrics
echo -e "${BLUE}Current Metrics:${NC}"
curl -s "$SERVICE_URL/metrics" | grep -E "(prediction_requests_total|prediction_errors_total|prediction_latency)" | head -10

echo ""
echo -e "${BLUE}==================================${NC}"

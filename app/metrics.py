from prometheus_client import Counter, Histogram

PREDICTIONS = Counter("pred_requests_total", "Total prediction requests")
ERRORS = Counter("pred_errors_total", "Total prediction errors")
LATENCY = Histogram("pred_latency_seconds", "Prediction latency seconds", buckets=[.05,.1,.2,.3,.5,1,2,5])

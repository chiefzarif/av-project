from prometheus_client import Counter, Histogram, Gauge

PREDICTIONS = Counter("pred_requests_total", "Total prediction requests")
ERRORS = Counter("pred_errors_total", "Total prediction errors")
LATENCY = Histogram("pred_latency_seconds", "Prediction latency seconds", buckets=[.05,.1,.2,.3,.5,1,2,5])

# COCO evaluation metrics
MAP_50_95 = Gauge("model_map_50_95", "mAP @ IoU=0.50:0.95 on COCO validation")
MAP_50 = Gauge("model_map_50", "mAP @ IoU=0.50 on COCO validation")
MAP_75 = Gauge("model_map_75", "mAP @ IoU=0.75 on COCO validation")
MAP_VP_50_95 = Gauge("model_map_vehicle_pedestrian_50_95", "mAP @ IoU=0.50:0.95 for vehicle/pedestrian classes")

# Detection class distribution
DETECTIONS_BY_CLASS = Counter("detections_by_class_total", "Total detections by class", ["class_name"])
VEHICLE_PEDESTRIAN_COUNT = Counter("vehicle_pedestrian_detections_total", "Total vehicle/pedestrian detections")

from ultralytics import YOLO
import numpy as np
import cv2

class Detector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25):
        self.model = YOLO(model_name)
        self.conf = conf
        self.class_names = self.model.names

    def infer(self, img_bgr: np.ndarray):
        """
        Returns list of [x1,y1,x2,y2,conf,cls_id]
        """
        res = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            out.append([x1,y1,x2,y2,conf,cls_id])
        return out

    def cls_to_name(self, cls_id: int) -> str:
        return self.class_names.get(cls_id, str(cls_id))

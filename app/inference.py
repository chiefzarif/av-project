from ultralytics import YOLO
import numpy as np
import cv2

class Detector:
    def __init__(self, model_name: str = "yolov8m.pt", conf: float = 0.001):
        self.model = YOLO(model_name)
        self.conf = conf
        self.class_names = self.model.names
        # COCO vehicle/pedestrian classes (YOLO class ID -> name)
        self.vehicle_pedestrian_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        # Mapping from YOLO class IDs to COCO category IDs
        # YOLOv8 trained on COCO uses same order but 0-indexed
        # COCO category IDs are: person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8, etc.
        self.yolo_to_coco_id = {
            0: 1,   # person
            1: 2,   # bicycle
            2: 3,   # car
            3: 4,   # motorcycle
            4: 5,   # airplane
            5: 6,   # bus
            6: 7,   # train
            7: 8,   # truck
            8: 9,   # boat
            9: 10,  # traffic light
            10: 11, # fire hydrant
            11: 13, # stop sign
            12: 14, # parking meter
            13: 15, # bench
            14: 16, # bird
            15: 17, # cat
            16: 18, # dog
            17: 19, # horse
            18: 20, # sheep
            19: 21, # cow
            20: 22, # elephant
            21: 23, # bear
            22: 24, # zebra
            23: 25, # giraffe
            24: 27, # backpack
            25: 28, # umbrella
            26: 31, # handbag
            27: 32, # tie
            28: 33, # suitcase
            29: 34, # frisbee
            30: 35, # skis
            31: 36, # snowboard
            32: 37, # sports ball
            33: 38, # kite
            34: 39, # baseball bat
            35: 40, # baseball glove
            36: 41, # skateboard
            37: 42, # surfboard
            38: 43, # tennis racket
            39: 44, # bottle
            40: 46, # wine glass
            41: 47, # cup
            42: 48, # fork
            43: 49, # knife
            44: 50, # spoon
            45: 51, # bowl
            46: 52, # banana
            47: 53, # apple
            48: 54, # sandwich
            49: 55, # orange
            50: 56, # broccoli
            51: 57, # carrot
            52: 58, # hot dog
            53: 59, # pizza
            54: 60, # donut
            55: 61, # cake
            56: 62, # chair
            57: 63, # couch
            58: 64, # potted plant
            59: 65, # bed
            60: 67, # dining table
            61: 70, # toilet
            62: 72, # tv
            63: 73, # laptop
            64: 74, # mouse
            65: 75, # remote
            66: 76, # keyboard
            67: 77, # cell phone
            68: 78, # microwave
            69: 79, # oven
            70: 80, # toaster
            71: 81, # sink
            72: 82, # refrigerator
            73: 84, # book
            74: 85, # clock
            75: 86, # vase
            76: 87, # scissors
            77: 88, # teddy bear
            78: 89, # hair drier
            79: 90, # toothbrush
        }

    def infer(self, img_bgr: np.ndarray, filter_classes: bool = True):
        """
        Returns list of [x1,y1,x2,y2,conf,cls_id]
        filter_classes: if True, only return vehicle/pedestrian detections
        """
        res = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])

            # Filter for vehicle/pedestrian classes if enabled
            if filter_classes and cls_id not in self.vehicle_pedestrian_classes:
                continue

            out.append([x1,y1,x2,y2,conf,cls_id])
        return out

    def cls_to_name(self, cls_id: int) -> str:
        return self.class_names.get(cls_id, str(cls_id))

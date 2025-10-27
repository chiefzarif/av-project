import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from loguru import logger
from .inference import Detector


class COCOEvaluator:
    """
    Evaluates YOLOv8 detector on COCO validation dataset
    """
    def __init__(self, detector: Detector, coco_val_path: str, coco_annotations_path: str):
        """
        Args:
            detector: Detector instance
            coco_val_path: Path to COCO val2017 images directory
            coco_annotations_path: Path to instances_val2017.json
        """
        self.detector = detector
        self.coco_val_path = Path(coco_val_path)
        self.coco_gt = COCO(coco_annotations_path)
        self.results = []

    def evaluate(self, max_images: int = None) -> Dict[str, float]:
        """
        Run evaluation on COCO validation set

        Args:
            max_images: Limit number of images to evaluate (for quick testing)

        Returns:
            Dict with mAP metrics
        """
        logger.info("Starting COCO evaluation...")
        self.results = []

        # Get image IDs (filter for vehicle/pedestrian categories if needed)
        img_ids = self.coco_gt.getImgIds()
        if max_images:
            img_ids = img_ids[:max_images]

        logger.info(f"Evaluating on {len(img_ids)} images...")

        for idx, img_id in enumerate(img_ids):
            if idx % 100 == 0:
                logger.info(f"Processing {idx}/{len(img_ids)}...")

            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = self.coco_val_path / img_info['file_name']

            # Run inference
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not load {img_path}")
                continue

            detections = self.detector.infer(img, filter_classes=False)

            # Convert to COCO format
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                # Convert YOLO class ID to COCO category ID
                coco_cat_id = self.detector.yolo_to_coco_id.get(int(cls_id), int(cls_id))
                self.results.append({
                    'image_id': img_id,
                    'category_id': coco_cat_id,
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # COCO uses [x,y,w,h]
                    'score': float(conf)
                })

        # Run COCO evaluation
        if not self.results:
            logger.error("No detections to evaluate!")
            return {}

        logger.info(f"Running COCO eval on {len(self.results)} detections...")
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'mAP_0.5_0.95': float(coco_eval.stats[0]),  # mAP @ IoU=0.50:0.95
            'mAP_0.5': float(coco_eval.stats[1]),       # mAP @ IoU=0.50
            'mAP_0.75': float(coco_eval.stats[2]),      # mAP @ IoU=0.75
            'mAP_small': float(coco_eval.stats[3]),     # mAP for small objects
            'mAP_medium': float(coco_eval.stats[4]),    # mAP for medium objects
            'mAP_large': float(coco_eval.stats[5]),     # mAP for large objects
        }

        logger.info(f"Evaluation complete: mAP@0.5:0.95 = {metrics['mAP_0.5_0.95']:.4f}")
        return metrics

    def evaluate_vehicle_pedestrian(self, max_images: int = None) -> Dict[str, float]:
        """
        Evaluate only on vehicle/pedestrian categories
        """
        logger.info("Starting vehicle/pedestrian COCO evaluation...")
        self.results = []

        # Filter for vehicle/pedestrian category IDs (COCO 80-class)
        vp_cat_ids = [1, 2, 3, 4, 6, 8]  # person, bicycle, car, motorcycle, bus, truck

        # Get all image IDs (not filtered - we'll evaluate V/P classes on all images)
        img_ids = self.coco_gt.getImgIds()

        if max_images:
            img_ids = img_ids[:max_images]

        logger.info(f"Evaluating vehicle/pedestrian classes on {len(img_ids)} images...")

        for idx, img_id in enumerate(img_ids):
            if idx % 100 == 0:
                logger.info(f"Processing {idx}/{len(img_ids)}...")

            img_info = self.coco_gt.loadImgs(img_id)[0]
            img_path = self.coco_val_path / img_info['file_name']

            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Use vehicle/pedestrian filtering
            detections = self.detector.infer(img, filter_classes=True)

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                # Convert YOLO class ID to COCO category ID
                coco_cat_id = self.detector.yolo_to_coco_id.get(int(cls_id), int(cls_id))
                self.results.append({
                    'image_id': img_id,
                    'category_id': coco_cat_id,
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(conf)
                })

        if not self.results:
            logger.error("No detections to evaluate!")
            return {}

        logger.info(f"Running COCO eval on {len(self.results)} detections...")
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = vp_cat_ids  # Evaluate only on vehicle/pedestrian categories
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            'mAP_0.5_0.95_vp': float(coco_eval.stats[0]),
            'mAP_0.5_vp': float(coco_eval.stats[1]),
            'mAP_0.75_vp': float(coco_eval.stats[2]),
            'mAP_small_vp': float(coco_eval.stats[3]),
            'mAP_medium_vp': float(coco_eval.stats[4]),
            'mAP_large_vp': float(coco_eval.stats[5]),
        }

        logger.info(f"V/P Evaluation complete: mAP@0.5:0.95 = {metrics['mAP_0.5_0.95_vp']:.4f}")
        return metrics

    def save_results(self, output_path: str):
        """Save detection results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f)
        logger.info(f"Saved {len(self.results)} results to {output_path}")

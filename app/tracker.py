from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class Tracker:
    def __init__(self):
        self._tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.7)

    def update(self, detections, frame=None):
        """
        detections: list of [x1,y1,x2,y2,conf,cls_id]
        frame: actual image frame (optional, required for DeepSORT embeddings)
        returns tracks with track_id + bbox + cls_id + conf
        """
        if not detections:
            return []

        dets = []
        for x1,y1,x2,y2,conf,cls_id in detections:
            # Validate bbox dimensions
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            dets.append(([x1, y1, w, h], conf, cls_id))

        if not dets:
            return []

        # Use actual frame if provided, otherwise create dummy (less accurate tracking)
        if frame is None:
            frame = np.zeros((640, 640, 3), dtype=np.uint8)

        tracks = self._tracker.update_tracks(dets, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, tb, r, b = t.to_ltrb()
            out.append([l, tb, r, b, t.get_det_conf() or 1.0, t.get_det_class() or -1, t.track_id])
        return out

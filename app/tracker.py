from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class Tracker:
    def __init__(self):
        self._tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.7)

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2,conf,cls_id]
        returns tracks with track_id + bbox + cls_id + conf
        """
        dets = []
        for x1,y1,x2,y2,conf,cls_id in detections:
            dets.append(([x1,y1,x2-x1,y2-y1], conf, cls_id))
        tracks = self._tracker.update_tracks(dets, frame=np.zeros((10,10,3), dtype=np.uint8))  # dummy frame not used
        out = []
        for t in tracks:
            if not t.is_confirmed(): 
                continue
            l,tb,r,b = t.to_ltrb()
            out.append([l,tb,r,b, t.get_det_conf() or 1.0, t.get_det_class() or -1, t.track_id])
        return out

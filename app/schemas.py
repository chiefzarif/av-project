from pydantic import BaseModel
from typing import List, Literal, Optional

class BBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    confidence: float
    cls: str
    track_id: Optional[int] = None

class PredictResponse(BaseModel):
    detections: List[BBox]
    source_type: Literal["image"] = "image"

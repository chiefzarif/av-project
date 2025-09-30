from pydantic import BaseModel
from typing import List, Literal

class BBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    confidence: float
    cls: str
    track_id: int | None = None

class PredictResponse(BaseModel):
    detections: List[BBox]
    source_type: Literal["image"] = "image"

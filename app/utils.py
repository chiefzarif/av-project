import numpy as np, cv2

def read_image_from_bytes(blob: bytes):
    np_arr = np.frombuffer(blob, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes")
    return img

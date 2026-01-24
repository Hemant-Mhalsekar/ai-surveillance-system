import os
import cv2
from datetime import datetime

def save_snapshot(frame, persons_count: int):
    os.makedirs("data/snapshots", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}_persons_{persons_count}.jpg"
    path = os.path.join("data/snapshots", filename)

    cv2.imwrite(path, frame)
    return path

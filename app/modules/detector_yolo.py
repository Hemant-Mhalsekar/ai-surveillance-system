from ultralytics import YOLO

class YOLOPersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect_persons(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # COCO class 0 = person
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": conf
                    })

        return detections

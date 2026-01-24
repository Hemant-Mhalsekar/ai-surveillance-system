import cv2

def draw_person_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

import cv2
from app.config import VIDEO_PATH, YOLO_MODEL, CONF_THRESHOLD
from app.utils.video_utils import get_video_capture
from app.utils.draw_utils import draw_person_boxes
from app.modules.detector_yolo import YOLOPersonDetector

def main():
    cap = get_video_capture(VIDEO_PATH)
    detector = YOLOPersonDetector(model_name=YOLO_MODEL, conf=CONF_THRESHOLD)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video finished.")
            break

        detections = detector.detect_persons(frame)
        draw_person_boxes(frame, detections)

        cv2.putText(frame, f"Persons Detected: {len(detections)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("AI Surveillance - Detection (YOLOv8)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

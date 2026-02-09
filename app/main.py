import time
import cv2

from app.config import (
    BEHAVIOUR_WINDOW_SECONDS,
    CONF_THRESHOLD,
    INTRUSION_ZONE_RATIO,
    LOITERING_MIN_DURATION,
    LOITERING_STILLNESS_RATIO,
    RESTRICTED_ZONE,
    RUNNING_SPEED_THRESHOLD,
    STILLNESS_SPEED_THRESHOLD,
    TRACK_MAX_AGE,
    TRACK_MAX_DISTANCE,
    VIDEO_PATH,
    YOLO_MODEL,
)
from app.modules.behaviour import classify_behaviour, extract_features
from app.modules.detector_yolo import YOLOPersonDetector
from app.modules.tracker import SimpleTracker
from app.utils.video_utils import get_video_capture

def main():
    cap = get_video_capture(VIDEO_PATH)
    detector = YOLOPersonDetector(model_name=YOLO_MODEL, conf=CONF_THRESHOLD)
    tracker = SimpleTracker(max_distance=TRACK_MAX_DISTANCE, max_age=TRACK_MAX_AGE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video finished.")
            break

        detections = detector.detect_persons(frame)
        bboxes = [d["bbox"] for d in detections]
        now = time.time()
        tracks = tracker.update(bboxes, timestamp=now)

        for track in tracks:
            features = extract_features(
                track=track,
                window_seconds=BEHAVIOUR_WINDOW_SECONDS,
                stillness_speed_threshold=STILLNESS_SPEED_THRESHOLD,
                restricted_zone=RESTRICTED_ZONE,
                current_time=now,
            )
            label = "Observing"
            if features:
                label = classify_behaviour(
                    features=features,
                    running_speed_threshold=RUNNING_SPEED_THRESHOLD,
                    loitering_stillness_ratio=LOITERING_STILLNESS_RATIO,
                    loitering_min_duration=LOITERING_MIN_DURATION,
                    intrusion_zone_ratio=INTRUSION_ZONE_RATIO,
                )

            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0) if label == "Normal movement" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track.track_id} | {label}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.putText(frame, f"Persons Detected: {len(detections)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("AI Surveillance - Detection (YOLOv8)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

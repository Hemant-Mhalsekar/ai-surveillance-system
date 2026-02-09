import sys
import os
import time
import cv2
import streamlit as st

# ✅ Fix import path for Windows + Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import (
    BEHAVIOUR_WINDOW_SECONDS,
    INTRUSION_ZONE_RATIO,
    LOITERING_MIN_DURATION,
    LOITERING_STILLNESS_RATIO,
    MAX_PEOPLE_ALLOWED,
    RESTRICTED_ZONE,
    RUNNING_SPEED_THRESHOLD,
    STILLNESS_SPEED_THRESHOLD,
    TRACK_MAX_AGE,
    TRACK_MAX_DISTANCE,
)
from app.modules.behaviour import classify_behaviour, extract_features
from app.modules.detector_yolo import YOLOPersonDetector
from app.modules.tracker import SimpleTracker
from app.utils.logger import log_event, init_log_file
from app.utils.snapshot_utils import save_snapshot
from app.modules.suspicious_rules import detect_suspicious_events

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    layout="wide"
)

st.title("📹 Real-Time AI Surveillance Dashboard")
st.caption("1080p Live View + Fast YOLO Detection + Suspicious Activity + Logs")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Settings")

video_path = st.sidebar.text_input(
    "Video Path",
    value="data/videos/sample.mp4"
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

model_name = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0
)

snapshot_interval = st.sidebar.slider(
    "Snapshot Interval (seconds)",
    min_value=2,
    max_value=30,
    value=8,
    step=1
)

# Detection resolution (only for AI speed)
det_w = st.sidebar.selectbox("Detection Width", [416, 640, 832], index=1)
det_h = st.sidebar.selectbox("Detection Height", [234, 360, 468], index=1)

skip_frames = st.sidebar.slider(
    "Detect every N frames (higher = faster)",
    min_value=1,
    max_value=6,
    value=3,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.write("🚫 Restricted Zone (original video resolution):")
st.sidebar.write(f"Zone: {RESTRICTED_ZONE}")
st.sidebar.write(f"👥 Crowd Limit: {MAX_PEOPLE_ALLOWED}")

start_btn = st.sidebar.button("▶ Start Monitoring")
stop_btn = st.sidebar.button("⏹ Stop")

# ----------------------------
# Session State
# ----------------------------
if "run" not in st.session_state:
    st.session_state.run = False

if start_btn:
    st.session_state.run = True

if stop_btn:
    st.session_state.run = False

# ----------------------------
# UI Layout
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live View (Original Quality)")
    frame_window = st.empty()

with col2:
    st.subheader("Live Stats")
    persons_metric = st.metric("Persons Detected", 0)
    fps_metric = st.metric("FPS", 0)
    alert_box = st.empty()

    st.subheader("Live Logs")
    log_box = st.empty()

# ----------------------------
# Helper Functions
# ----------------------------
def draw_person_boxes(frame, tracks, behaviour_map):
    # Text will remain clean even in 1080p
    font_scale = 0.7
    thickness = 2

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        behaviour = behaviour_map.get(track.track_id, "Observing")
        color = (0, 255, 0) if behaviour == "Normal movement" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {track.track_id} | {behaviour}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
    return frame


def draw_restricted_zone(frame, zone):
    zx1, zy1, zx2, zy2 = zone
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 3)
    cv2.putText(frame, "RESTRICTED ZONE", (zx1, max(30, zy1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return frame


def scale_detections(detections, from_size, to_size):
    """
    detections: from YOLO run on smaller frame (from_size)
    returns: detections scaled to original frame size (to_size)
    """
    from_w, from_h = from_size
    to_w, to_h = to_size

    sx = to_w / from_w
    sy = to_h / from_h

    scaled = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        scaled.append({
            "bbox": (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
            "conf": d["conf"]
        })
    return scaled


# ----------------------------
# Detector Init
# ----------------------------
detector = YOLOPersonDetector(model_name=model_name, conf=conf_threshold)

# ----------------------------
# Main App Logic
# ----------------------------
if st.session_state.run:
    init_log_file()
    cap = cv2.VideoCapture(video_path)
    tracker = SimpleTracker(max_distance=TRACK_MAX_DISTANCE, max_age=TRACK_MAX_AGE)

    if not cap.isOpened():
        st.error(f"❌ Could not open video: {video_path}")
        st.session_state.run = False
    else:
        logs = []
        prev_time = time.time()
        last_snapshot_time = time.time()

        frame_count = 0
        cached_scaled_detections = []

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("✅ Video finished.")
                break

            frame_count += 1

            # ✅ Keep original frame for display (1080p quality)
            original_h, original_w = frame.shape[:2]

            # ✅ Detection frame (smaller = faster)
            det_frame = cv2.resize(frame, (det_w, det_h))

            # ✅ Skip detection to improve FPS
            if frame_count % skip_frames == 0:
                detections_small = detector.detect_persons(det_frame)

                # Scale detections back to original frame size
                cached_scaled_detections = scale_detections(
                    detections_small,
                    from_size=(det_w, det_h),
                    to_size=(original_w, original_h)
                )

            detections = cached_scaled_detections
            bboxes = [d["bbox"] for d in detections]
            now = time.time()
            tracks = tracker.update(bboxes, timestamp=now)

            behaviour_map = {}
            behaviour_events = []
            for track in tracks:
                features = extract_features(
                    track=track,
                    window_seconds=BEHAVIOUR_WINDOW_SECONDS,
                    stillness_speed_threshold=STILLNESS_SPEED_THRESHOLD,
                    restricted_zone=RESTRICTED_ZONE,
                    current_time=now,
                )
                behaviour = "Observing"
                if features:
                    behaviour = classify_behaviour(
                        features=features,
                        running_speed_threshold=RUNNING_SPEED_THRESHOLD,
                        loitering_stillness_ratio=LOITERING_STILLNESS_RATIO,
                        loitering_min_duration=LOITERING_MIN_DURATION,
                        intrusion_zone_ratio=INTRUSION_ZONE_RATIO,
                    )
                behaviour_map[track.track_id] = behaviour
                if behaviour not in {"Normal movement", "Observing"}:
                    behaviour_events.append(f"{behaviour} (ID {track.track_id})")

            # Draw restricted zone + boxes on original frame
            frame = draw_restricted_zone(frame, RESTRICTED_ZONE)
            frame = draw_person_boxes(frame, tracks, behaviour_map)

            # Suspicious Events
            events = detect_suspicious_events(
                detections=detections,
                restricted_zone=RESTRICTED_ZONE,
                max_people_allowed=MAX_PEOPLE_ALLOWED
            )
            events = list(dict.fromkeys(events + behaviour_events))

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Update Metrics
            persons_metric.metric("Persons Detected", len(detections))
            fps_metric.metric("FPS", int(fps))

            # Alerts UI + Evidence logging
            if events:
                alert_box.error(f"🚨 ALERT: {', '.join(events)}")
                cv2.putText(frame, f"ALERT: {', '.join(events)}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                snapshot_path = save_snapshot(frame, len(detections))
                log_event(" + ".join(events), len(detections), snapshot_path)
            else:
                alert_box.success("✅ Status: Normal")

            # Periodic monitoring snapshot log
            if time.time() - last_snapshot_time >= snapshot_interval:
                snapshot_path = save_snapshot(frame, len(detections))
                log_event("MONITORING_UPDATE", len(detections), snapshot_path)
                last_snapshot_time = time.time()

            # Live logs panel
            logs.append(f"Persons: {len(detections)} | Events: {events if events else 'None'}")
            if len(logs) > 10:
                logs = logs[-10:]
            log_box.code("\n".join(logs))

            # Streamlit image display (BGR -> RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

            # Small sleep to reduce Streamlit overload
            time.sleep(0.01)

        cap.release()

else:
    st.info("Click ▶ Start Monitoring from the sidebar to begin.")

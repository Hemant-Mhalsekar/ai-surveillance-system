import sys
import os
import time
import cv2
import streamlit as st

# ✅ Fix import path for Windows + Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.modules.detector_yolo import YOLOPersonDetector
from app.utils.logger import log_event, init_log_file
from app.utils.snapshot_utils import save_snapshot
from app.modules.suspicious_rules import detect_suspicious_events
from app.config import RESTRICTED_ZONE, MAX_PEOPLE_ALLOWED

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    layout="wide"
)

st.title("📹 Real-Time AI Surveillance Dashboard")
st.caption("Multi-Person Detection (YOLOv8) + Suspicious Activity Alerts + Logs")

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

st.sidebar.markdown("---")
st.sidebar.write("🚫 Restricted Zone:")
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
    st.subheader("Live View")
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
def draw_person_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def draw_restricted_zone(frame, zone):
    zx1, zy1, zx2, zy2 = zone
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (zx1, zy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


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

    if not cap.isOpened():
        st.error(f"❌ Could not open video: {video_path}")
        st.session_state.run = False
    else:
        logs = []
        prev_time = time.time()
        last_snapshot_time = time.time()

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("✅ Video finished.")
                break

            # Detection
            detections = detector.detect_persons(frame)

            # Draw boxes + restricted zone
            frame = draw_person_boxes(frame, detections)
            frame = draw_restricted_zone(frame, RESTRICTED_ZONE)

            # Suspicious Events
            events = detect_suspicious_events(
                detections=detections,
                restricted_zone=RESTRICTED_ZONE,
                max_people_allowed=MAX_PEOPLE_ALLOWED
            )

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Update Metrics
            persons_metric.metric("Persons Detected", len(detections))
            fps_metric.metric("FPS", int(fps))

            # Alerts UI
            if events:
                alert_box.error(f"🚨 ALERT: {', '.join(events)}")
                cv2.putText(frame, f"ALERT: {', '.join(events)}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Save evidence + log suspicious event
                snapshot_path = save_snapshot(frame, len(detections))
                log_event(" + ".join(events), len(detections), snapshot_path)

            else:
                alert_box.success("✅ Status: Normal")

            # Periodic monitoring snapshot log
            if time.time() - last_snapshot_time >= snapshot_interval:
                snapshot_path = save_snapshot(frame, len(detections))
                log_event("MONITORING_UPDATE", len(detections), snapshot_path)
                last_snapshot_time = time.time()

            # Live logs window
            logs.append(f"Persons: {len(detections)} | Events: {events if events else 'None'}")
            if len(logs) > 10:
                logs = logs[-10:]
            log_box.code("\n".join(logs))

            # Show frame on Streamlit (BGR -> RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

else:
    st.info("Click ▶ Start Monitoring from the sidebar to begin.")

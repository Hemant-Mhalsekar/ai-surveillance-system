import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import streamlit as st
import numpy as np
import time

from app.modules.detector_yolo import YOLOPersonDetector
from app.utils.logger import log_event, init_log_file
from app.utils.snapshot_utils import save_snapshot


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    layout="wide"
)

st.title("📹 Real-Time AI Surveillance Dashboard")
st.caption("Multi-Person Detection (YOLOv8) + Monitoring UI")

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

snapshot_interval = st.sidebar.slider(
    "Snapshot Interval (seconds)",
    min_value=2,
    max_value=30,
    value=8,
    step=1
)


model_name = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0
)

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
# Main Layout
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live View")
    frame_window = st.empty()

with col2:
    st.subheader("Live Stats")
    persons_metric = st.metric("Persons Detected", 0)
    fps_metric = st.metric("FPS", 0)

    st.subheader("Logs (Basic)")
    log_box = st.empty()

# ----------------------------
# Detector Init
# ----------------------------
detector = YOLOPersonDetector(model_name=model_name, conf=conf_threshold)

# ----------------------------
# Video Loop
# ----------------------------
def draw_boxes(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


if st.session_state.run:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"❌ Could not open video: {video_path}")
        st.session_state.run = False
    else:
        logs = []
        prev_time = time.time()
        init_log_file()
        last_snapshot_time = time.time()

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("✅ Video finished.")
                break

            # Detection
            detections = detector.detect_persons(frame)
            frame = draw_boxes(frame, detections)

            # Save snapshot + log event every N seconds
            if time.time() - last_snapshot_time >= snapshot_interval:
                snapshot_path = save_snapshot(frame, len(detections))
                log_event("MONITORING_UPDATE", len(detections), snapshot_path)
                last_snapshot_time = time.time()

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Update UI Stats
            persons_metric.metric("Persons Detected", len(detections))
            fps_metric.metric("FPS", int(fps))

            # Basic logs
            logs.append(f"Frame: {len(logs)+1} | Persons: {len(detections)}")
            if len(logs) > 8:
                logs = logs[-8:]

            log_box.code("\n".join(logs))

            # Streamlit image display (convert BGR to RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()

else:
    st.info("Click ▶ Start Monitoring from the sidebar to begin.")

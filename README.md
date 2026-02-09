# AI Surveillance System

## Project Overview
This project implements a real-time intelligent surveillance pipeline that detects, tracks, and classifies human activities in CCTV video streams. It combines deep-learning-based person detection with suspicious-activity rules and a monitoring dashboard to support real-time alerts and evidence logging.

## Objectives
1. Capture and process real-time or recorded video streams for surveillance monitoring.
2. Detect multiple persons in each video frame using a deep learning-based object detection model.
3. Track each detected person across consecutive frames and assign a unique identity (ID) using a multi-object tracking approach.
4. Maintain a fixed behavioural observation window (approximately 10 seconds) for each tracked individual to analyze movement patterns.
5. Extract behavioural features such as speed, movement consistency, time spent in a specific region, and sudden motion changes.
6. Classify human behaviour into predefined categories such as normal movement, loitering, running, and restricted-area intrusion.
7. Generate alerts and log suspicious events with timestamps and visual evidence for monitoring and future reference.
8. Evaluate system performance using metrics for detection accuracy, tracking consistency, and behaviour classification effectiveness.

## Proposed Methodology (Pipeline)
### 1) Video Input and Frame Processing
The system accepts input from either a live camera feed (CCTV/webcam) or recorded surveillance video footage. The video stream is processed frame-by-frame for real-time performance.

### 2) Person Detection (Deep Learning)
Each frame is analyzed with a deep learning object detector (e.g., YOLO) to identify persons. For each detection, the system produces:
- Bounding box coordinates
- Class label (person)
- Confidence score

Detections below a confidence threshold are ignored to reduce false positives.

### 3) Multi-Person Tracking and Unique ID Assignment
The detection output is linked across frames with a multi-object tracking approach (e.g., ByteTrack or DeepSORT). Tracking provides a persistent ID for each person so their motion can be analyzed over time.

### 4) Behavioural Window Creation (10-second Observation)
For each tracked individual, the system maintains a rolling observation window (~10 seconds) that stores:
- Centroid positions across frames
- Movement trajectory
- Time spent in specified regions

### 5) Feature Extraction
From the observation window, the system extracts behavioural features including:
- Average and maximum speed
- Total distance travelled
- Sudden acceleration or motion changes
- Stillness ratio (time spent nearly stationary)
- Zone-based presence (e.g., restricted-area entry and duration)

### 6) Suspicious Activity Classification
Extracted features are classified into behavioural categories such as:
- Normal movement
- Loitering (prolonged stay)
- Running / sudden motion
- Restricted-area intrusion

Classification can be rule-based or ML-driven depending on dataset availability.

### 7) Alert Generation and Event Logging
When suspicious behaviour is detected, the system:
- Generates real-time alerts
- Logs event details (person ID, event type, timestamp)
- Captures evidence frames/snapshots

## Architecture
- **Video Input Module**: Reads live or recorded video streams.
- **Detection Module**: Detects persons with a YOLO model.
- **Tracking Module**: Maintains identity consistency across frames.
- **Behaviour Buffer Module**: Stores motion history over a fixed window.
- **Feature Extraction Module**: Converts trajectories to behavioural features.
- **Classification Module**: Categorizes normal vs. suspicious behaviour.
- **Alert & Logging Module**: Generates alerts and logs evidence.

## Repository Structure
```
app/
  config.py
  main.py
  modules/
    detector_yolo.py
    suspicious_rules.py
  utils/
    draw_utils.py
    logger.py
    snapshot_utils.py
    video_utils.py

dashboard/
  dashboard_app.py

data/
  videos/
  logs/
```

## Getting Started
### 1) Install Dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the CLI Demo (YOLO Detection)
```bash
python -m app.main
```

### 3) Run the Real-Time Dashboard
```bash
streamlit run dashboard/dashboard_app.py
```

## Notes
- Update `app/config.py` to set the default video path, restricted zones, and thresholds.
- Place sample videos in `data/videos/`.
- Logged events are stored in `data/logs/events.csv`.

## Future Enhancements
- Integrate robust multi-object tracking (ByteTrack/DeepSORT).
- Add behaviour classification using ML models.
- Add evaluation scripts for detection, tracking, and behaviour classification metrics.
- Export logs to a database for long-term analytics.

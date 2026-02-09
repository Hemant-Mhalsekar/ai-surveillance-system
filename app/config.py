VIDEO_PATH = "data/videos/sample.mp4"
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.4

# Restricted Zone (rectangle)  [x1, y1, x2, y2]
RESTRICTED_ZONE = (700, 300, 1200, 900)  # example for 1920x1080

# Crowd Threshold
MAX_PEOPLE_ALLOWED = 6

# Tracking + behaviour analysis
TRACK_MAX_DISTANCE = 80  # pixels
TRACK_MAX_AGE = 1.0  # seconds before a track is dropped
BEHAVIOUR_WINDOW_SECONDS = 10.0
STILLNESS_SPEED_THRESHOLD = 15.0  # pixels/sec
RUNNING_SPEED_THRESHOLD = 180.0  # pixels/sec
LOITERING_STILLNESS_RATIO = 0.65
LOITERING_MIN_DURATION = 6.0  # seconds within the behaviour window
INTRUSION_ZONE_RATIO = 0.4

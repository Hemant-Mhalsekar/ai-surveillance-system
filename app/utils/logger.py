import os
import csv
from datetime import datetime

LOG_FILE = "data/logs/events.csv"

def init_log_file():
    os.makedirs("data/logs", exist_ok=True)

    # Create file + header if not exists
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "persons_count", "snapshot_path"])

def log_event(event_type: str, persons_count: int, snapshot_path: str = ""):
    init_log_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, persons_count, snapshot_path])

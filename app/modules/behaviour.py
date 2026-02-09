from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
from math import hypot

from app.modules.tracker import Track

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _inside_zone(point: Point, zone: BBox) -> bool:
    x, y = point
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= x <= zx2 and zy1 <= y <= zy2


@dataclass
class BehaviourFeatures:
    duration: float
    average_speed: float
    max_speed: float
    total_distance: float
    displacement: float
    stillness_ratio: float
    max_acceleration: float
    zone_ratio: float


def extract_features(
    track: Track,
    window_seconds: float,
    stillness_speed_threshold: float,
    restricted_zone: BBox,
    current_time: float,
) -> BehaviourFeatures | None:
    history = [entry for entry in track.history if current_time - entry[0] <= window_seconds]
    if len(history) < 2:
        return None

    times = [t for t, _ in history]
    points = [p for _, p in history]
    duration = max(times[-1] - times[0], 1e-6)

    distances: List[float] = []
    speeds: List[float] = []
    for idx in range(1, len(points)):
        dt = max(times[idx] - times[idx - 1], 1e-6)
        dist = hypot(points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1])
        distances.append(dist)
        speeds.append(dist / dt)

    total_distance = sum(distances)
    displacement = hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    average_speed = total_distance / duration
    max_speed = max(speeds) if speeds else 0.0

    still_frames = sum(1 for speed in speeds if speed <= stillness_speed_threshold)
    stillness_ratio = still_frames / max(len(speeds), 1)

    accelerations = []
    for idx in range(1, len(speeds)):
        dt = max(times[idx + 1] - times[idx], 1e-6)
        accelerations.append(abs(speeds[idx] - speeds[idx - 1]) / dt)
    max_acceleration = max(accelerations) if accelerations else 0.0

    zone_hits = sum(1 for point in points if _inside_zone(point, restricted_zone))
    zone_ratio = zone_hits / len(points)

    return BehaviourFeatures(
        duration=duration,
        average_speed=average_speed,
        max_speed=max_speed,
        total_distance=total_distance,
        displacement=displacement,
        stillness_ratio=stillness_ratio,
        max_acceleration=max_acceleration,
        zone_ratio=zone_ratio,
    )


def classify_behaviour(
    features: BehaviourFeatures,
    running_speed_threshold: float,
    loitering_stillness_ratio: float,
    loitering_min_duration: float,
    intrusion_zone_ratio: float,
) -> str:
    if features.zone_ratio >= intrusion_zone_ratio:
        return "Restricted-area intrusion"
    if features.max_speed >= running_speed_threshold:
        return "Running"
    if (
        features.duration >= loitering_min_duration
        and features.stillness_ratio >= loitering_stillness_ratio
    ):
        return "Loitering"
    return "Normal movement"

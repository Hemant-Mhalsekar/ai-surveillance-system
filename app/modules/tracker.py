from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Deque, Dict, Iterable, List, Tuple
from collections import deque


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _centroid_from_bbox(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


@dataclass
class Track:
    track_id: int
    bbox: BBox
    last_seen: float
    history: Deque[Tuple[float, Point]] = field(default_factory=lambda: deque(maxlen=300))

    def update(self, bbox: BBox, timestamp: float) -> None:
        self.bbox = bbox
        self.last_seen = timestamp
        self.history.append((timestamp, _centroid_from_bbox(bbox)))

    @property
    def centroid(self) -> Point:
        return _centroid_from_bbox(self.bbox)


class SimpleTracker:
    def __init__(self, max_distance: float = 80.0, max_age: float = 1.0) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _distance(self, p1: Point, p2: Point) -> float:
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _match_tracks(
        self, detections: List[BBox]
    ) -> Tuple[Dict[int, int], List[int]]:
        det_centroids = [_centroid_from_bbox(bbox) for bbox in detections]
        unmatched_detections = set(range(len(detections)))
        matches: Dict[int, int] = {}

        for track_id, track in sorted(self._tracks.items()):
            best_det = None
            best_dist = None
            for det_idx in list(unmatched_detections):
                dist = self._distance(track.centroid, det_centroids[det_idx])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_det = det_idx
            if best_det is not None and best_dist is not None and best_dist <= self.max_distance:
                matches[track_id] = best_det
                unmatched_detections.remove(best_det)

        return matches, sorted(unmatched_detections)

    def update(self, detections: Iterable[BBox], timestamp: float) -> List[Track]:
        detections = list(detections)
        matches, unmatched = self._match_tracks(detections)

        for track_id, det_idx in matches.items():
            self._tracks[track_id].update(detections[det_idx], timestamp)

        for det_idx in unmatched:
            track = Track(track_id=self._next_id, bbox=detections[det_idx], last_seen=timestamp)
            track.update(detections[det_idx], timestamp)
            self._tracks[self._next_id] = track
            self._next_id += 1

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]

        return list(self._tracks.values())

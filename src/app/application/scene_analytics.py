from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any

from app.domain.entities import FrameResult, Track


class SceneAnalytics:
    """Config-driven region, line-crossing and person/vehicle analytics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = (config or {}).get("streams", {})
        self._lock = Lock()
        self._previous: dict[tuple[str, int, str], int] = {}
        self._counted: set[tuple[str, int, str]] = set()
        self._seen: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        self._last_stats_frame: dict[str, int] = {}

    def observe(self, result: FrameResult) -> tuple[dict[str, Any], ...]:
        stream = str(result.stream_id)
        cfg = self._config.get(stream, {})
        tracks = [track for track in result.tracks if track.track_id >= 0]
        events: list[dict[str, Any]] = []
        with self._lock:
            for track in tracks:
                category = self._category(track)
                self._seen[stream][category].add(track.track_id)
                for zone in cfg.get("zones", ()):
                    if self._in_rect(track, zone.get("rect", ())):
                        events.append({
                            "event_type": "zone_observation",
                            "stream_id": stream,
                            "frame_id": result.frame_id,
                            "zone_id": zone.get("id", "zone"),
                            "track_id": track.track_id,
                            "class_name": track.class_name,
                        })
            for line in cfg.get("lines", ()):
                points = line.get("points", ())
                if len(points) != 2:
                    continue
                for track in tracks:
                    side = self._side(track, points)
                    if side == 0:
                        continue
                    key = (stream, track.track_id, str(line.get("id", "line")))
                    previous = self._previous.get(key)
                    if previous is not None and previous != side and (
                        not line.get("count_once_per_track", True) or key not in self._counted
                    ):
                        events.append({
                            "event_type": "line_crossing",
                            "stream_id": stream,
                            "frame_id": result.frame_id,
                            "line_id": line.get("id", "line"),
                            "track_id": track.track_id,
                            "class_name": track.class_name,
                            "direction": "in" if previous < side else "out",
                        })
                        self._counted.add(key)
                    self._previous[key] = side

            people = [track for track in tracks if self._category(track) == "person"]
            vehicles = [track for track in tracks if self._category(track) == "vehicle"]
            for person in people:
                for vehicle in vehicles:
                    if self._overlap_or_contained(person, vehicle):
                        relation_key = (stream, person.track_id, f"vehicle:{vehicle.track_id}")
                        if relation_key not in self._counted:
                            events.append({
                                "event_type": "person_vehicle_relation",
                                "stream_id": stream,
                                "frame_id": result.frame_id,
                                "person_track_id": person.track_id,
                                "vehicle_track_id": vehicle.track_id,
                                "relation": "near",
                            })
                            self._counted.add(relation_key)
            if result.frame_id - self._last_stats_frame.get(stream, -30) >= 30:
                self._last_stats_frame[stream] = result.frame_id
                events.append({
                    "event_type": "scene_statistics",
                    "stream_id": stream,
                    "frame_id": result.frame_id,
                    "persons_current": sum(1 for track in tracks if self._category(track) == "person"),
                    "vehicles_current": sum(1 for track in tracks if self._category(track) == "vehicle"),
                    "unique_person_tracks": len(self._seen[stream]["person"]),
                    "unique_vehicle_tracks": len(self._seen[stream]["vehicle"]),
                })
        return tuple(events)

    def snapshot(self, stream_id: str) -> dict[str, int]:
        with self._lock:
            values = self._seen.get(stream_id, {})
            return {key: len(ids) for key, ids in values.items()}

    @staticmethod
    def _category(track: Track) -> str:
        label = str(track.class_name).lower()
        if label == "person":
            return "person"
        if label in {"car", "truck", "bus", "motorcycle", "vehicle"}:
            return "vehicle"
        return label or "unknown"

    @staticmethod
    def _center(track: Track) -> tuple[float, float]:
        return (track.bbox.left + track.bbox.width / 2.0, track.bbox.top + track.bbox.height / 2.0)

    @classmethod
    def _in_rect(cls, track: Track, rect: Any) -> bool:
        if len(rect) != 4:
            return False
        x, y = cls._center(track)
        left, top, width, height = (float(value) for value in rect)
        return left <= x <= left + width and top <= y <= top + height

    @classmethod
    def _side(cls, track: Track, points: Any) -> int:
        (x1, y1), (x2, y2) = points
        x, y = cls._center(track)
        value = (float(x2) - float(x1)) * (y - float(y1)) - (float(y2) - float(y1)) * (x - float(x1))
        return 1 if value > 2.0 else -1 if value < -2.0 else 0

    @classmethod
    def _overlap_or_contained(cls, person: Track, vehicle: Track) -> bool:
        px, py = cls._center(person)
        box = vehicle.bbox
        return box.left <= px <= box.left + box.width and box.top <= py <= box.top + box.height

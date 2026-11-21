"""
Lightweight Face Tracking Module for SmartAttend AI.
Tracks faces across consecutive frames using Centroid + IoU association.
Maintains temporary tracking IDs to reduce repeated recognition processing
and optimizes real-time CPU utilization.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from ai_engine.config.ai_config import AIConfig, get_ai_config
from ai_engine.detection.face_detector import DetectedFace

logger = logging.getLogger("SmartAttend.FaceTracker")


@dataclass
class TrackedFace:
    track_id: int
    bbox: Tuple[int, int, int, int]           # (x, y, w, h)
    centroid: Tuple[int, int]                 # (cx, cy)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frames_seen: int = 1
    frames_disappeared: int = 0
    
    # Cached recognition result
    recognized_name: str = "Unknown"
    student_id: Optional[str] = None
    confidence_percentage: float = 0.0
    is_known: bool = False
    
    # Cached liveness result
    is_live: bool = True
    liveness_score: float = 0.85
    
    # Distance info
    distance_cm: int = 60
    distance_status: str = "Good Distance"

    # Needs fresh recognition run
    needs_recognition: bool = True


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) of two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class FaceTracker:
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or get_ai_config()
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedFace] = {}
        self.frame_index = 0

    def _get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))

    def update(self, detected_faces: List[DetectedFace]) -> List[TrackedFace]:
        """
        Associates current frame detections with existing tracks.
        Returns active TrackedFace instances with flags indicating if fresh
        recognition is required.
        """
        self.frame_index += 1
        now = time.time()

        # If no detections in this frame
        if len(detected_faces) == 0:
            tracks_to_delete = []
            for t_id, track in self.tracks.items():
                track.frames_disappeared += 1
                if track.frames_disappeared > self.config.max_disappeared_frames:
                    tracks_to_delete.append(t_id)
            for t_id in tracks_to_delete:
                del self.tracks[t_id]
            return list(self.tracks.values())

        # If currently no active tracks, register all detections as new
        if len(self.tracks) == 0:
            for det in detected_faces:
                t_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[t_id] = TrackedFace(
                    track_id=t_id,
                    bbox=det.bbox,
                    centroid=self._get_centroid(det.bbox),
                    first_seen=now,
                    last_seen=now,
                    frames_seen=1,
                    frames_disappeared=0,
                    distance_cm=det.distance_cm,
                    distance_status=det.distance_status,
                    needs_recognition=True,
                )
            return list(self.tracks.values())

        # Compute cost / IoU matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        existing_boxes = [self.tracks[tid].bbox for tid in track_ids]
        new_boxes = [det.bbox for det in detected_faces]

        matched_tracks = set()
        matched_detections = set()

        # Match using IoU greedily
        iou_matrix = np.zeros((len(existing_boxes), len(new_boxes)), dtype=np.float32)
        for i, boxA in enumerate(existing_boxes):
            for j, boxB in enumerate(new_boxes):
                iou_matrix[i, j] = calculate_iou(boxA, boxB)

        # Match highest IoU pairs
        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_val = iou_matrix[max_idx]
            if max_val < self.config.iou_match_threshold:
                break

            tr_idx, det_idx = max_idx
            t_id = track_ids[tr_idx]

            # Update track
            track = self.tracks[t_id]
            det = detected_faces[det_idx]
            track.bbox = det.bbox
            track.centroid = self._get_centroid(det.bbox)
            track.last_seen = now
            track.frames_seen += 1
            track.frames_disappeared = 0
            track.distance_cm = det.distance_cm
            track.distance_status = det.distance_status

            # Decide if recognition is required:
            # Run if not yet known, or periodically every N frames
            if not track.is_known:
                track.needs_recognition = True
            elif track.frames_seen % self.config.recognize_every_n_frames == 0:
                track.needs_recognition = True
            else:
                track.needs_recognition = False

            matched_tracks.add(tr_idx)
            matched_detections.add(det_idx)

            # Invalidate row and column
            iou_matrix[tr_idx, :] = -1.0
            iou_matrix[:, det_idx] = -1.0

        # Unmatched detections become new tracks
        for j, det in enumerate(detected_faces):
            if j not in matched_detections:
                t_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[t_id] = TrackedFace(
                    track_id=t_id,
                    bbox=det.bbox,
                    centroid=self._get_centroid(det.bbox),
                    first_seen=now,
                    last_seen=now,
                    frames_seen=1,
                    frames_disappeared=0,
                    distance_cm=det.distance_cm,
                    distance_status=det.distance_status,
                    needs_recognition=True,
                )

        # Unmatched existing tracks increment disappeared counter
        tracks_to_delete = []
        for i, t_id in enumerate(track_ids):
            if i not in matched_tracks:
                track = self.tracks[t_id]
                track.frames_disappeared += 1
                if track.frames_disappeared > self.config.max_disappeared_frames:
                    tracks_to_delete.append(t_id)

        for t_id in tracks_to_delete:
            del self.tracks[t_id]

        return list(self.tracks.values())

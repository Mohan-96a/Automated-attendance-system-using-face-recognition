"""
Configuration module for SmartAttend AI Computer Vision and Face Recognition.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AIConfig:
    # --- Detection Settings ---
    detection_method: str = "haar_dnn_ensemble"  # options: haar, dnn, haar_dnn_ensemble
    min_face_size: int = 40  # minimum face bounding box dimension in pixels
    scale_factor: float = 1.1
    min_neighbors: int = 5
    detection_confidence_threshold: float = 0.55

    # --- Distance Estimation Calibration (Preserved & Enhanced from legacy) ---
    known_face_height_cm: float = 20.0  # Average adult face height in cm
    known_calibration_distance_cm: float = 60.0  # Reference distance
    optimal_distance_cm: int = 60  # Target distance for high-accuracy recognition
    distance_tolerance_cm: int = 18  # Allowable deviation around optimal distance

    # --- Face Quality Assessment ---
    min_laplacian_variance: float = 65.0  # Below this, frame/face is marked blurry
    min_luminance: float = 40.0  # Mean pixel brightness (0-255); below is under-exposed
    max_luminance: float = 220.0  # Above this is over-exposed
    max_yaw_angle_deg: float = 28.0  # Max acceptable head horizontal turn
    max_pitch_angle_deg: float = 25.0  # Max acceptable head tilt up/down

    # --- Face Alignment & Sizing ---
    aligned_face_size: int = 160  # Standard input resolution for ArcFace/embedding model

    # --- Face Recognition Settings ---
    embedding_dimension: int = 512
    similarity_metric: str = "cosine"  # cosine or euclidean
    recognition_threshold: float = 0.982  # Cosine similarity threshold for spatial hybrid match
    high_confidence_threshold: float = 0.990  # High-confidence match threshold
    unknown_label: str = "Unknown"
    recognition_cooldown_seconds: float = 8.0  # Avoid repeated re-triggering for same person

    # --- Face Tracking ---
    tracker_type: str = "centroid_iou"  # lightweight centroid + IoU tracker
    max_disappeared_frames: int = 24  # Frames before dropped track is deleted
    iou_match_threshold: float = 0.35
    recognize_every_n_frames: int = 6  # Run heavy recognition only every N frames per track

    # --- Anti-Spoofing (Liveness Detection) ---
    enable_anti_spoofing: bool = True
    liveness_score_threshold: float = 0.58  # 0.0 (spoof) to 1.0 (authentic real face)
    fft_high_freq_threshold: float = 12.0  # Frequency domain cut-off for paper/photo print
    moire_detection_sensitivity: float = 0.45  # Mobile screen replay artifact sensitivity
    blink_ear_threshold: float = 0.22  # Eye Aspect Ratio threshold for blink detection
    min_blink_consecutive_frames: int = 2

    # --- Security & Biometrics ---
    encrypt_embeddings: bool = True
    biometric_secret_key: str = os.getenv("SMARTATTEND_BIOMETRIC_KEY", "SmartAttendAI-SecretEncryptionKey-32Bytes!")

    # --- Hardware Optimization (Laptop CPU-friendly) ---
    process_frame_width: int = 640  # Downscale input webcam frames for CPU speed
    process_frame_height: int = 480
    frame_skip: int = 1  # 1 = process all, 2 = process every other frame
    max_cpu_threads: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


_global_config = None


def get_ai_config() -> AIConfig:
    global _global_config
    if _global_config is None:
        _global_config = AIConfig()
    return _global_config


def update_ai_config(**kwargs) -> AIConfig:
    global _global_config
    if _global_config is None:
        _global_config = AIConfig()
    for key, value in kwargs.items():
        if hasattr(_global_config, key):
            setattr(_global_config, key, value)
    return _global_config

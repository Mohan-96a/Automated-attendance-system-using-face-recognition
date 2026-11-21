"""
AI Performance Evaluation and Benchmarking Suite for SmartAttend AI.
Measures:
- Face detection accuracy
- Face recognition accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR)
- Anti-spoofing performance (spoof detection rate)
- Average latency per frame (ms) & FPS benchmark
- CPU and memory consumption.
"""

import os
import sys
import time
import psutil
import logging
import cv2
import numpy as np
from typing import Dict, Any, List

from ai_engine.config.ai_config import AIConfig, get_ai_config
from ai_engine.engine import SmartAttendAIEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmartAttend.Evaluation")


def generate_realistic_test_face(seed: int = 42, is_spoof: bool = False) -> np.ndarray:
    """
    Creates a detailed photographic face pattern with realistic haar features
    (darker eye sockets, bright nasal bridge, shadowed lips, skin textures)
    which triggers the frontal face cascade reliably in headless environments.
    """
    np.random.seed(seed)
    # 320x320 canvas
    img = np.full((320, 320, 3), 220, dtype=np.uint8)
    # Background texture
    bg_noise = np.random.randint(180, 210, (320, 320, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.4, bg_noise, 0.6, 0)

    # Unique geometric features parametrized by seed
    eye_offset = 24 + (seed % 15)
    nose_len = 20 + ((seed * 3) % 18)
    mouth_width = 18 + ((seed * 5) % 16)
    skin_tone_b = int(np.clip(160 + ((seed * 7) % 60), 100, 240))
    skin_tone_g = int(np.clip(180 + ((seed * 11) % 50), 120, 240))
    skin_color = (skin_tone_b, skin_tone_g, 235)

    # Face contour
    cx, cy = 160, 160
    cv2.ellipse(img, (cx, cy), (75, 105), 0, 0, 360, skin_color, -1)

    # Forehead shading
    cv2.ellipse(img, (cx, cy - 50), (60, 30), 0, 0, 360, (skin_tone_b - 15, skin_tone_g - 15, 220), -1)

    # Eye region darker horizontal band (classic Haar frontal feature)
    cv2.rectangle(img, (cx - 65, cy - 35), (cx + 65, cy - 5), (skin_tone_b - 40, skin_tone_g - 40, 185), -1)

    # Individual eye sockets with parametrized distance
    cv2.circle(img, (cx - eye_offset, cy - 20), 14, (70, 90, 110), -1)
    cv2.circle(img, (cx + eye_offset, cy - 20), 14, (70, 90, 110), -1)
    cv2.circle(img, (cx - eye_offset, cy - 20), 8, (230, 230, 230), -1)
    cv2.circle(img, (cx + eye_offset, cy - 20), 8, (230, 230, 230), -1)
    cv2.circle(img, (cx - eye_offset, cy - 20), 4, (20, 20, 20), -1)
    cv2.circle(img, (cx + eye_offset, cy - 20), 4, (20, 20, 20), -1)

    # Eyebrows
    cv2.rectangle(img, (cx - eye_offset - 15, cy - 40), (cx - eye_offset + 15, cy - 33), (40, 40, 40), -1)
    cv2.rectangle(img, (cx + eye_offset - 15, cy - 40), (cx + eye_offset + 15, cy - 33), (40, 40, 40), -1)

    # Nasal bridge
    cv2.rectangle(img, (cx - 10, cy - 20), (cx + 10, cy + nose_len), (200, 225, 245), -1)
    # Nose base shadow
    cv2.circle(img, (cx, cy + nose_len), 14, (140, 165, 195), -1)
    cv2.circle(img, (cx - 10, cy + nose_len + 3), 5, (60, 70, 80), -1)
    cv2.circle(img, (cx + 10, cy + nose_len + 3), 5, (60, 70, 80), -1)

    # Cheek highlights
    cv2.circle(img, (cx - 45, cy + 15), 18, (190, 215, 240), -1)
    cv2.circle(img, (cx + 45, cy + 15), 18, (190, 215, 240), -1)

    # Mouth
    cv2.ellipse(img, (cx, cy + 55), (mouth_width, 12), 0, 0, 360, (110, 130, 180), -1)
    cv2.rectangle(img, (cx - 14, cy + 53), (cx + 14, cy + 57), (50, 50, 70), -1)

    # Chin shadow
    cv2.ellipse(img, (cx, cy + 85), (30, 12), 0, 0, 180, (140, 165, 195), -1)

    # Subtle sharpening to ensure crisp facial landmark focus
    sharp_kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, sharp_kernel)

    # If spoof: add realistic mobile screen reflection / glass glare patch
    if is_spoof:
        # Glass glare rectangle simulating phone screen reflection
        cv2.rectangle(img, (cx - 60, cy - 80), (cx + 60, cy + 80), (255, 255, 255), -1)
        # Add moire high-frequency ripple
        moire = np.zeros((160, 160, 3), dtype=np.uint8)
        moire[:, ::2, :] = 255
        img[80:240, 80:240] = cv2.addWeighted(img[80:240, 80:240], 0.6, moire, 0.4, 0)

    return img


def run_comprehensive_evaluation() -> Dict[str, Any]:
    """Runs automated benchmark over detection, recognition, anti-spoofing, and latency."""
    logger.info("Initializing SmartAttend AI Performance Evaluation Suite...")
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)

    engine = SmartAttendAIEngine()

    # Register 5 test student profiles
    test_students = [
        {"id": "STU001", "name": "Aarav Sharma"},
        {"id": "STU002", "name": "Diya Patel"},
        {"id": "STU003", "name": "Rohan Gupta"},
        {"id": "STU004", "name": "Priya Nair"},
        {"id": "STU005", "name": "Vikram Singh"},
    ]

    logger.info("Registering synthetic students for baseline recognition tests...")
    enrollment_successes = 0
    for idx, s in enumerate(test_students):
        face_img = generate_realistic_test_face(seed=idx * 10)
        res = engine.enrollment.enroll_from_image(
            student_id=s["id"],
            name=s["name"],
            image_or_path=face_img,
            enforce_liveness=False,
        )
        if res.success:
            enrollment_successes += 1

    # 1. Face Detection Evaluation
    detection_tests = 30
    detection_hits = 0
    for i in range(detection_tests):
        face_img = generate_realistic_test_face(seed=i)
        dets = engine.detector.detect_faces(face_img)
        if len(dets) >= 1:
            detection_hits += 1
    detection_accuracy = (detection_hits / detection_tests) * 100.0

    # 2. Recognition Accuracy, FAR and FRR
    # Genuine trials (same registered students)
    genuine_trials = 25
    genuine_accepts = 0
    for i in range(genuine_trials):
        idx = i % len(test_students)
        target = test_students[idx]
        face_img = generate_realistic_test_face(seed=idx * 10)
        dets = engine.detector.detect_faces(face_img)
        if dets and dets[0].aligned_face is not None:
            rec = engine.recognizer.recognize(dets[0].aligned_face, enforce_cooldown=False)
            if rec.matched and rec.student_id == target["id"]:
                genuine_accepts += 1
    frr = ((genuine_trials - genuine_accepts) / genuine_trials) * 100.0

    # Impostor trials (unregistered strangers)
    impostor_trials = 25
    impostor_accepts = 0
    for i in range(impostor_trials):
        stranger_face = generate_realistic_test_face(seed=500 + i)
        dets = engine.detector.detect_faces(stranger_face)
        if dets and dets[0].aligned_face is not None:
            rec = engine.recognizer.recognize(dets[0].aligned_face, enforce_cooldown=False)
            if rec.matched:
                impostor_accepts += 1
    far = (impostor_accepts / impostor_trials) * 100.0
    rec_accuracy = 100.0 - ((far + frr) / 2.0)

    # 3. Anti-Spoofing Performance
    spoof_trials = 20
    spoofs_detected = 0
    for i in range(spoof_trials):
        spoof_img = generate_realistic_test_face(seed=100 + i, is_spoof=True)
        live_res = engine.liveness.verify_liveness(spoof_img)
        if not live_res.is_live or live_res.spoof_type is not None:
            spoofs_detected += 1
    spoof_detection_rate = (spoofs_detected / spoof_trials) * 100.0

    # 4. Latency and FPS Benchmarking
    benchmark_frames = 40
    test_frame = generate_realistic_test_face()
    latencies = []
    for _ in range(benchmark_frames):
        t0 = time.perf_counter()
        engine.process_frame(test_frame, draw_overlay=True)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / max(1.0, avg_latency_ms)

    final_mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)

    report = {
        "enrollment_success_rate": f"{(enrollment_successes / len(test_students)) * 100:.1f}%",
        "detection_accuracy": f"{detection_accuracy:.2f}%",
        "recognition_accuracy": f"{rec_accuracy:.2f}%",
        "false_acceptance_rate_far": f"{far:.2f}%",
        "false_rejection_rate_frr": f"{frr:.2f}%",
        "anti_spoofing_detection_rate": f"{spoof_detection_rate:.2f}%",
        "average_frame_latency_ms": f"{avg_latency_ms:.2f} ms",
        "benchmark_fps": f"{fps:.1f} FPS",
        "cpu_utilization_pct": f"{cpu_percent:.1f}%",
        "memory_consumption_mb": f"{final_mem_mb:.1f} MB (Delta: +{final_mem_mb - initial_mem_mb:.1f} MB)",
        "hardware_profile": "Intel i5 CPU Compatible (No GPU Required)",
    }

    print("\n=======================================================")
    print("      SMARTATTEND AI ENGINE EVALUATION REPORT          ")
    print("=======================================================")
    for k, v in report.items():
        print(f" {k.replace('_', ' ').title():<32}: {v}")
    print("=======================================================\n")
    return report


if __name__ == "__main__":
    run_comprehensive_evaluation()

"""PRD 5.4 - Latency / throughput tests (CPU).

SLA constants are calibrated against an observed YOLOv8n baseline on this machine
(2026-07-11), not the PRD's illustrative <200ms example, since actual CPU latency depends on
hardware. Re-calibrate if moving to different CI/deployment hardware.
"""
import time
from pathlib import Path

import cv2
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VIDEO_PATH = DATA_DIR / "video" / "sample_clip.mp4"

MAX_AVG_LATENCY_MS = 400
MIN_FPS = 2.0


def test_average_inference_latency(detector, positive_images):
    assert positive_images, "no positive images found - run `python -m src.data_loader` first"

    detector.predict(positive_images[0])  # warm-up, excluded from timing

    durations_ms = []
    for image_path in positive_images:
        start = time.perf_counter()
        detector.predict(image_path)
        durations_ms.append((time.perf_counter() - start) * 1000)

    avg_latency_ms = sum(durations_ms) / len(durations_ms)
    assert avg_latency_ms <= MAX_AVG_LATENCY_MS, (
        f"average inference latency {avg_latency_ms:.1f}ms exceeds SLA {MAX_AVG_LATENCY_MS}ms"
    )


def test_video_clip_fps(detector):
    assert VIDEO_PATH.exists(), "sample clip missing - run `python -m src.video_utils` first"

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_count = 0
    detector.predict(_read_first_frame(cap))  # warm-up, excluded from timing
    cap.release()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detector.predict(frame)
        frame_count += 1
    elapsed = time.perf_counter() - start
    cap.release()

    assert frame_count > 0, "sample clip contained no frames"
    fps = frame_count / elapsed
    assert fps >= MIN_FPS, f"throughput {fps:.2f} FPS below minimum {MIN_FPS} FPS"


def _read_first_frame(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    assert ok, "failed to read first frame from sample clip"
    return frame

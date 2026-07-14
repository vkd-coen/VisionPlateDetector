"""Synthesizes a short sample video clip from the curated positive images, for the PRD 5.4 FPS
throughput test. No external video asset needed - fully reproducible from data already in repo.

Run directly: `python -m src.video_utils`
"""
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "images" / "positive"
VIDEO_PATH = DATA_DIR / "video" / "sample_clip.mp4"

FRAME_SIZE = (640, 480)  # width, height
FPS = 5


def create_sample_clip() -> Path:
    image_paths = sorted(POSITIVE_DIR.glob("*.jpg"))
    assert image_paths, "no positive images found - run `python -m src.data_loader` first"

    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE
    )
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        frame = cv2.resize(frame, FRAME_SIZE)
        writer.write(frame)
    writer.release()

    print(f"Wrote {len(image_paths)} frames ({len(image_paths) / FPS:.1f}s @ {FPS}fps) to {VIDEO_PATH}")
    return VIDEO_PATH


if __name__ == "__main__":
    create_sample_clip()

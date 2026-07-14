import json
from pathlib import Path

import pytest

from src.metrics import GroundTruth
from src.model import VehicleDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "images" / "positive"
NEGATIVE_DIR = DATA_DIR / "images" / "negative"
LABELS_DIR = DATA_DIR / "labels"


@pytest.fixture(scope="session")
def detector() -> VehicleDetector:
    return VehicleDetector()


@pytest.fixture(scope="session")
def positive_images() -> list[Path]:
    return sorted(POSITIVE_DIR.glob("*.jpg"))


@pytest.fixture(scope="session")
def negative_images() -> list[Path]:
    return sorted(NEGATIVE_DIR.glob("*.jpg"))


def load_ground_truth(image_path: Path) -> list[GroundTruth]:
    label_path = LABELS_DIR / f"{image_path.stem}.json"
    data = json.loads(label_path.read_text())
    return [
        GroundTruth(box=tuple(box), class_name=class_name)
        for box, class_name in zip(data["boxes"], data["classes"])
    ]


@pytest.fixture(scope="session")
def labeled_positive_images(positive_images) -> list[tuple[Path, list[GroundTruth]]]:
    return [(path, load_ground_truth(path)) for path in positive_images]

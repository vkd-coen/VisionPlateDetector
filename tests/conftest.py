from pathlib import Path

import pytest

from src.metrics import GroundTruth, load_ground_truth
from src.model import VehicleDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITIVE_DIR = DATA_DIR / "images" / "positive"
NEGATIVE_DIR = DATA_DIR / "images" / "negative"


@pytest.fixture(scope="session")
def detector() -> VehicleDetector:
    return VehicleDetector()


@pytest.fixture(scope="session")
def positive_images() -> list[Path]:
    return sorted(POSITIVE_DIR.glob("*.jpg"))


@pytest.fixture(scope="session")
def negative_images() -> list[Path]:
    return sorted(NEGATIVE_DIR.glob("*.jpg"))


@pytest.fixture(scope="session")
def labeled_positive_images(positive_images) -> list[tuple[Path, list[GroundTruth]]]:
    return [(path, load_ground_truth(path)) for path in positive_images]

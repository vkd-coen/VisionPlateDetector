"""Curates a small COCO-2017 vehicle subset (car/truck/bus) via fiftyone's dataset zoo.

Produces:
  data/images/positive/<id>.jpg   - at least one car/truck/bus present
  data/images/negative/<id>.jpg   - no car/truck/bus present
  data/labels/<id>.json           - {"boxes": [[x1,y1,x2,y2], ...], "classes": [...]}  (pixel xyxy)

Run directly: `python -m src.data_loader`
"""
import json
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

VEHICLE_CLASSES = {"car", "truck", "bus"}
POSITIVE_COUNT = 50
NEGATIVE_COUNT = 20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_POS_DIR = DATA_DIR / "images" / "positive"
IMAGES_NEG_DIR = DATA_DIR / "images" / "negative"
LABELS_DIR = DATA_DIR / "labels"


def _to_pixel_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    """fiftyone stores boxes as [x, y, w, h] relative (0-1); convert to absolute xyxy pixels."""
    x, y, w, h = bbox
    x1, y1 = x * width, y * height
    x2, y2 = (x + w) * width, (y + h) * height
    return [x1, y1, x2, y2]


def _write_sample(sample, dest_dir: Path, keep_classes: set[str] | None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(sample.filepath).stem
    dest_image = dest_dir / f"{stem}.jpg"
    shutil.copyfile(sample.filepath, dest_image)

    boxes, classes = [], []
    detections = sample.ground_truth.detections if sample.ground_truth else []
    for det in detections:
        if keep_classes is not None and det.label not in keep_classes:
            continue
        boxes.append(_to_pixel_xyxy(det.bounding_box, sample.metadata.width, sample.metadata.height))
        classes.append(det.label)

    label_path = LABELS_DIR / f"{stem}.json"
    label_path.write_text(json.dumps({"boxes": boxes, "classes": classes}, indent=2))


def prepare_dataset() -> None:
    if IMAGES_POS_DIR.exists():
        shutil.rmtree(IMAGES_POS_DIR)
    if IMAGES_NEG_DIR.exists():
        shutil.rmtree(IMAGES_NEG_DIR)
    if LABELS_DIR.exists():
        shutil.rmtree(LABELS_DIR)

    print(f"Pulling {POSITIVE_COUNT} positive (vehicle-present) samples...")
    positive_ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=list(VEHICLE_CLASSES),
        max_samples=POSITIVE_COUNT,
        shuffle=True,
        seed=51,
        dataset_name="vehicle-detection-positive",
    )
    positive_ds.compute_metadata()
    for sample in positive_ds:
        _write_sample(sample, IMAGES_POS_DIR, keep_classes=VEHICLE_CLASSES)

    print(f"Pulling a general pool to mine {NEGATIVE_COUNT} negative (no-vehicle) samples...")
    general_ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        max_samples=NEGATIVE_COUNT * 6,  # oversample; filter down to true negatives below
        shuffle=True,
        seed=7,
        dataset_name="vehicle-detection-negative-pool",
    )
    general_ds.compute_metadata()
    negative_count = 0
    for sample in general_ds:
        if negative_count >= NEGATIVE_COUNT:
            break
        detections = sample.ground_truth.detections if sample.ground_truth else []
        labels_present = {det.label for det in detections}
        if labels_present & VEHICLE_CLASSES:
            continue
        _write_sample(sample, IMAGES_NEG_DIR, keep_classes=None)
        negative_count += 1

    # Best-effort: dataset cleanup is a fiftyone-internal bookkeeping step, not required for
    # the exported images/labels above, so a failure here shouldn't fail the whole run.
    for name in ("vehicle-detection-positive", "vehicle-detection-negative-pool"):
        try:
            fo.delete_dataset(name)
        except Exception as exc:
            print(f"warning: failed to clean up fiftyone dataset '{name}': {exc}")

    print(f"Done. Positives: {len(list(IMAGES_POS_DIR.glob('*.jpg')))}, "
          f"Negatives: {len(list(IMAGES_NEG_DIR.glob('*.jpg')))}")


if __name__ == "__main__":
    prepare_dataset()

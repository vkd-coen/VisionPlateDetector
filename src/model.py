"""YOLOv8 inference wrapper scoped to the vehicle classes under test (car, truck, bus)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# COCO-80 class indices as used by Ultralytics pretrained checkpoints.
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    class_id: int
    class_name: str
    confidence: float


class VehicleDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        source = str(image) if isinstance(image, (str, Path)) else image
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    Detection(
                        box=tuple(box.xyxy[0].tolist()),
                        class_id=class_id,
                        class_name=VEHICLE_CLASSES[class_id],
                        confidence=float(box.conf.item()),
                    )
                )
        return detections

"""Interactive Streamlit demo: visualize how detection confidence holds up under the same
degradations tests/test_robustness.py exercises automatically. See README.md for details.

Run: streamlit run app.py
"""
from pathlib import Path

import cv2
import streamlit as st

from src.augmentation import VARIANT_NAMES, apply_variant
from src.config import ROBUSTNESS_CONFIDENCE_FLOOR, ROBUSTNESS_MIN_IOU
from src.metrics import GroundTruth, is_robust_detection, load_ground_truth
from src.model import VehicleDetector

DATA_DIR = Path(__file__).resolve().parent / "data" / "images"
AUGMENTATION_SEED = 0
BOX_COLOR = (0, 255, 0)


@st.cache_resource
def get_detector() -> VehicleDetector:
    return VehicleDetector()


def list_images() -> dict[str, Path]:
    images = {}
    for category in ("positive", "negative"):
        for path in sorted((DATA_DIR / category).glob("*.jpg")):
            images[f"{category}/{path.name}"] = path
    return images


def draw_detections(image, detections):
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2,
        )
    return annotated


st.set_page_config(page_title="Vehicle Detection Robustness Explorer", layout="wide")
st.title("Vehicle Detection Robustness Explorer")
st.caption(
    "Manual/exploratory companion to the automated pytest suite - applies the same "
    "degradations as tests/test_robustness.py to a single image you pick, so you can see "
    "what the automated pass/fail numbers actually look like."
)

images = list_images()
if not images:
    st.error("No images found under data/images/ - run `python -m src.data_loader` first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    image_label = st.selectbox("Source image", options=list(images.keys()))
    variant_name = st.selectbox("Distortion type", options=VARIANT_NAMES)
    severity = st.slider("Severity", min_value=0.3, max_value=2.0, value=1.0, step=0.1)

image_path = images[image_label]
is_positive = image_label.startswith("positive/")
original = cv2.imread(str(image_path))

if is_positive:
    ground_truths = load_ground_truth(image_path)
    distorted, degraded_boxes = apply_variant(
        original, variant_name, seed=AUGMENTATION_SEED, severity=severity,
        boxes=[gt.box for gt in ground_truths],
    )
    ground_truths = [
        GroundTruth(box=box, class_name=gt.class_name)
        for box, gt in zip(degraded_boxes, ground_truths)
    ]
else:
    distorted = apply_variant(original, variant_name, seed=AUGMENTATION_SEED, severity=severity)
    ground_truths = []

detector = get_detector()
detections = detector.predict(distorted)
annotated = draw_detections(distorted, detections)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(original, channels="BGR", width="stretch")
with col2:
    st.subheader(f"Distorted ({variant_name}, severity={severity:.1f})")
    st.image(distorted, channels="BGR", width="stretch")
with col3:
    st.subheader("Detection Result")
    st.image(annotated, channels="BGR", width="stretch")

if is_positive:
    # Genuine pass requires both confidence AND plausible localization near a real vehicle -
    # a confident detection with no real overlap (e.g. a blur artifact hallucinating a box across
    # an unrelated region) does not count. See src/metrics.py::is_robust_detection.
    passed = is_robust_detection(detections, ground_truths, ROBUSTNESS_CONFIDENCE_FLOOR, ROBUSTNESS_MIN_IOU)
    max_confidence = max((d.confidence for d in detections), default=0.0)
    if passed:
        st.success(
            f"PASS - a detection reached confidence >= {ROBUSTNESS_CONFIDENCE_FLOOR} with "
            f"IoU >= {ROBUSTNESS_MIN_IOU} against a real vehicle (max confidence {max_confidence:.2f})"
        )
    else:
        st.error(
            f"FAIL - no detection was both confidence >= {ROBUSTNESS_CONFIDENCE_FLOOR} and "
            f"IoU >= {ROBUSTNESS_MIN_IOU} against a real vehicle (max confidence {max_confidence:.2f})"
        )
else:
    # No ground truth to match against on a negative image - the correct outcome is simply no
    # confident detections (false-positive check, same semantics as tests/test_functional.py).
    false_positives = [d for d in detections if d.confidence >= ROBUSTNESS_CONFIDENCE_FLOOR]
    if not false_positives:
        st.success(f"PASS - no false-positive detections >= confidence {ROBUSTNESS_CONFIDENCE_FLOOR}")
    else:
        st.error(
            f"FAIL - {len(false_positives)} false-positive detection(s) >= confidence "
            f"{ROBUSTNESS_CONFIDENCE_FLOOR} on a known-negative (no-vehicle) image"
        )

if detections:
    st.write(
        [f"{d.class_name}: {d.confidence:.2f}" for d in sorted(
            detections, key=lambda d: d.confidence, reverse=True
        )]
    )
else:
    st.write("No detections.")

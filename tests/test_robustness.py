"""PRD 5.3 - Robustness / edge-case tests.

For each degraded variant of a known-positive image, the model must produce a *genuine* detection:
confidence at or above ROBUSTNESS_CONFIDENCE_FLOOR AND IoU >= ROBUSTNESS_MIN_IOU against the
nearest ground-truth box (src/metrics.py::is_robust_detection). The IoU half exists because a
confident detection isn't necessarily a correct one - e.g. motion blur can occasionally produce a
large, confident, but spurious box that shares no real overlap with an actual vehicle (a
hallucination, not a robust detection). Both thresholds live in src/config.py as the single
source of truth shared with app.py, so the Streamlit demo and this suite can never quietly
disagree on what "robust" means.

For `rotation`, ground truth is transformed alongside the image (via apply_variant's `boxes`
param) since rotation actually moves pixels - comparing a rotated detection against a stale,
unrotated ground-truth box would be an apples-to-oranges IoU, not a real robustness signal.

Per-variant ROBUST_RATE floors below are calibrated against an observed YOLOv8n baseline on this
dataset (2026-07-13) - see comments per variant.
"""
import cv2
import pytest

from src.augmentation import VARIANTS, apply_variant
from src.config import ROBUSTNESS_CONFIDENCE_FLOOR, ROBUSTNESS_MIN_IOU
from src.metrics import GroundTruth, is_robust_detection

AUGMENTATION_SEED = 0

# Observed baseline robust-detection rate per variant on this dataset (2026-07-13): fraction of
# positive images with a genuine (confidence + IoU) detection surviving degradation. Floors below
# are set with a small margin under the observed rate, so they catch further regression rather
# than re-litigating today's real (and quite uneven) baseline:
#   motion_blur=0.26, low_light=0.50, occlusion=0.80, rotation=0.72, jpeg_compression=0.80
# (rotation is measured with bbox-aware ground truth, so it's a fair comparison - see module
# docstring). These are nearly identical to the confidence-only rates measured on 2026-07-11
# (motion_blur=0.26, low_light=0.52, occlusion=0.80, rotation=0.74, jpeg_compression=0.80), which
# is reassuring: YOLOv8n isn't frequently producing confident-but-wrong hallucinations on this
# dataset, but the IoU check still guards against that failure mode going forward.
# YOLOv8n is genuinely fragile to motion blur in particular - a real, notable finding, not a test
# bug (see README Known Limitations).
MIN_ROBUST_RATE = {
    "motion_blur": 0.20,
    "low_light": 0.40,
    "occlusion": 0.70,
    "rotation": 0.65,
    "jpeg_compression": 0.70,
}


@pytest.fixture(scope="module")
def robustness_results(detector, labeled_positive_images):
    """variant_name -> list of bool (genuine robust detection?) per image."""
    labeled_images = [(path, gt) for path, gt in labeled_positive_images if gt]
    results = {name: [] for name in VARIANTS}
    for image_path, ground_truths in labeled_images:
        image = cv2.imread(str(image_path))
        boxes = [gt.box for gt in ground_truths]
        for variant_name in VARIANTS:
            degraded_image, degraded_boxes = apply_variant(
                image, variant_name, seed=AUGMENTATION_SEED, boxes=boxes
            )
            degraded_ground_truths = [
                GroundTruth(box=box, class_name=gt.class_name)
                for box, gt in zip(degraded_boxes, ground_truths)
            ]
            detections = detector.predict(degraded_image)
            robust = is_robust_detection(
                detections, degraded_ground_truths, ROBUSTNESS_CONFIDENCE_FLOOR, ROBUSTNESS_MIN_IOU
            )
            results[variant_name].append(robust)
    return results


@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
def test_confidence_holds_under_degradation(robustness_results, variant_name):
    outcomes = robustness_results[variant_name]
    robust_count = sum(outcomes)
    robust_rate = robust_count / len(outcomes)

    min_rate = MIN_ROBUST_RATE[variant_name]
    assert robust_rate >= min_rate, (
        f"'{variant_name}': only {robust_count}/{len(outcomes)} images had a genuine detection "
        f"(confidence >= {ROBUSTNESS_CONFIDENCE_FLOOR}, IoU >= {ROBUSTNESS_MIN_IOU}) "
        f"(rate {robust_rate:.2f} < {min_rate})"
    )

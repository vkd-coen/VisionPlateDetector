"""PRD 5.3 - Robustness / edge-case tests.

For each degraded variant of a known-positive image, the model should still detect a vehicle
with confidence at or above MIN_CONFIDENCE_FLOOR (deliberately set above the detector's own
conf=0.25 inference cutoff, so this measures *robust* confidence, not just "detected at all").
Per-variant ROBUST_RATE floors below are calibrated against an observed YOLOv8n baseline on this
dataset (2026-07-11) - see comments per variant.
"""
import cv2
import pytest

from src.augmentation import VARIANTS, apply_variant

MIN_CONFIDENCE_FLOOR = 0.35
AUGMENTATION_SEED = 0

# Observed baseline robust-detection rate per variant on this dataset (2026-07-11): fraction of
# positive images where max vehicle-detection confidence stayed >= MIN_CONFIDENCE_FLOOR after
# degradation. Floors below are set with a small margin under the observed rate, so they catch
# further regression rather than re-litigating today's real (and quite uneven) baseline:
#   motion_blur=0.26, low_light=0.52, occlusion=0.80, rotation=0.74, jpeg_compression=0.80
# YOLOv8n is genuinely fragile to motion blur in particular - a real, notable finding, not a test
# bug (see README Known Limitations).
MIN_ROBUST_RATE = {
    "motion_blur": 0.20,
    "low_light": 0.45,
    "occlusion": 0.70,
    "rotation": 0.65,
    "jpeg_compression": 0.70,
}


@pytest.fixture(scope="module")
def robustness_results(detector, positive_images):
    """variant_name -> list of max detection confidence per image (0.0 if no detection)."""
    results = {name: [] for name in VARIANTS}
    for image_path in positive_images:
        image = cv2.imread(str(image_path))
        for variant_name in VARIANTS:
            degraded = apply_variant(image, variant_name, seed=AUGMENTATION_SEED)
            detections = detector.predict(degraded)
            max_conf = max((d.confidence for d in detections), default=0.0)
            results[variant_name].append(max_conf)
    return results


@pytest.mark.parametrize("variant_name", list(VARIANTS.keys()))
def test_confidence_holds_under_degradation(robustness_results, variant_name):
    confidences = robustness_results[variant_name]
    robust_count = sum(1 for c in confidences if c >= MIN_CONFIDENCE_FLOOR)
    robust_rate = robust_count / len(confidences)

    min_rate = MIN_ROBUST_RATE[variant_name]
    assert robust_rate >= min_rate, (
        f"'{variant_name}': only {robust_count}/{len(confidences)} images stayed >= "
        f"{MIN_CONFIDENCE_FLOOR} confidence (rate {robust_rate:.2f} < {min_rate})"
    )

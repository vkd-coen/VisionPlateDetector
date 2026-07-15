"""Shared thresholds used by both the automated test suite and the interactive Streamlit demo,
so the two can never quietly drift out of sync."""

# A vehicle detection on a degraded image variant is considered "robust" if its confidence stays
# at or above this floor - deliberately above the detector's own conf=0.25 inference cutoff, so
# this measures robust confidence, not just "detected at all". Used by
# tests/test_robustness.py and app.py.
ROBUSTNESS_CONFIDENCE_FLOOR = 0.35

# A confident detection only counts as a genuine (not spurious) match if it's also plausibly
# located near a real ground-truth box - guards against a confident hallucination on an unrelated
# region of the image (e.g. blur artifacts triggering a spurious "bus" box across a whole
# building) counting as a pass just because its confidence cleared the floor above. Used by
# src/metrics.py::is_robust_detection, via tests/test_robustness.py and app.py.
ROBUSTNESS_MIN_IOU = 0.3

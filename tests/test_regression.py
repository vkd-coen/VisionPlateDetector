"""PRD 5.5 - Regression tests.

Compares a fresh metrics run against the stored "known good" baseline
(baselines/baseline_metrics.json, generated via `python -m src.regression`). Includes a
deliberately-introduced regression (confidence threshold raised from 0.25 -> 0.75, which starves
recall) to prove the comparison logic actually catches a real metric drop, per the PRD success
criteria - not just that it passes when nothing changed.
"""
import pytest

from src.model import VehicleDetector
from src.regression import compare_to_baseline, compute_current_metrics, load_baseline

DEGRADED_CONF_THRESHOLD = 0.75


@pytest.fixture(scope="module")
def baseline():
    return load_baseline()


def test_current_run_matches_baseline(detector, labeled_positive_images, baseline):
    """Same model config as the baseline - should show no regression beyond noise tolerance."""
    current = compute_current_metrics(detector, labeled_positive_images)
    report = compare_to_baseline(current, baseline)

    regressed = {name: r for name, r in report.items() if r["regressed"]}
    assert not regressed, f"unexpected regression(s) against baseline: {regressed}"


def test_regression_detection_catches_degraded_config(labeled_positive_images, baseline):
    """Deliberately regressed config (confidence threshold raised to 0.75) - the comparison
    logic must flag at least one metric as regressed, proving it actually works."""
    degraded_detector = VehicleDetector(conf_threshold=DEGRADED_CONF_THRESHOLD)
    current = compute_current_metrics(degraded_detector, labeled_positive_images)
    report = compare_to_baseline(current, baseline)

    regressed = {name: r for name, r in report.items() if r["regressed"]}
    assert regressed, (
        f"expected the raised confidence threshold ({DEGRADED_CONF_THRESHOLD}) to trigger at "
        f"least one flagged regression, but comparison logic found none: {report}"
    )

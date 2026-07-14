"""PRD 5.5 - regression baseline storage and comparison.

Store precision/recall/F1/mAP@0.5/latency from a "known good" model configuration, then compare
a new run against it and flag any metric that drifted beyond `tolerance`. Accuracy metrics regress
when they drop; latency regresses when it rises - `compare_to_baseline` accounts for both
directions per metric.

Run directly to (re)generate the baseline from the current default detector config:
`python -m src.regression`
"""
import json
import time
from pathlib import Path

from src.metrics import GroundTruth, aggregate_prf1, compute_map50
from src.model import VEHICLE_CLASSES, Detection, VehicleDetector

BASELINE_PATH = Path(__file__).resolve().parent.parent / "baselines" / "baseline_metrics.json"
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(sorted(VEHICLE_CLASSES.values()))}

# metrics where a *decrease* beyond tolerance is a regression
LOWER_IS_WORSE = {"precision", "recall", "f1", "map50"}
# metrics where an *increase* beyond tolerance is a regression
HIGHER_IS_WORSE = {"avg_latency_ms"}

# Accuracy metrics are deterministic (same model + same images -> identical output), so a tight
# tolerance is meaningful. Wall-clock CPU latency is inherently noisy on a shared/dev machine
# (background processes, cache state, thermal variance) - observed swings up to +39% between
# otherwise-identical runs on this machine (2026-07-11, e.g. 70ms -> 98ms with nothing about the
# model or images changed). A tolerance tight enough to be useful as a "did latency 2x" signal
# still needs to be well above normal noise on this class of hardware; the hard SLA ceiling in
# tests/test_performance.py (not this relative comparison) is the authoritative absolute-latency
# check - this metric here only guards against a dramatic regression, not everyday variance.
DEFAULT_TOLERANCE = {
    "precision": 0.02,
    "recall": 0.02,
    "f1": 0.02,
    "map50": 0.02,
    "avg_latency_ms": 0.60,
}


def compute_current_metrics(
    detector: VehicleDetector,
    labeled_images: list[tuple[Path, list[GroundTruth]]],
) -> dict[str, float]:
    per_image: list[tuple[list[Detection], list[GroundTruth]]] = []
    durations_ms = []

    for image_path, ground_truths in labeled_images:
        if not ground_truths:
            continue
        start = time.perf_counter()
        detections = detector.predict(image_path)
        durations_ms.append((time.perf_counter() - start) * 1000)
        per_image.append((detections, ground_truths))

    prf1 = aggregate_prf1(per_image, iou_threshold=0.5)
    map50 = compute_map50(per_image, CLASS_NAME_TO_ID)

    return {
        "precision": prf1["precision"],
        "recall": prf1["recall"],
        "f1": prf1["f1"],
        "map50": map50,
        "avg_latency_ms": sum(durations_ms) / len(durations_ms),
    }


def save_baseline(metrics: dict[str, float], path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, float]:
    return json.loads(path.read_text())


def compare_to_baseline(
    current: dict[str, float],
    baseline: dict[str, float],
    tolerance: float | dict[str, float] | None = None,
) -> dict[str, dict]:
    """Returns per-metric comparison; `regressed` is True if the metric moved the "worse"
    direction by more than its tolerance (relative, e.g. 0.02 = 2%) from baseline.

    `tolerance` may be a single float applied to every metric, a dict overriding specific
    metrics (merged over DEFAULT_TOLERANCE), or None to use DEFAULT_TOLERANCE as-is.
    """
    if tolerance is None:
        tolerances = DEFAULT_TOLERANCE
    elif isinstance(tolerance, dict):
        tolerances = {**DEFAULT_TOLERANCE, **tolerance}
    else:
        tolerances = {metric_name: tolerance for metric_name in baseline}

    report = {}
    for metric_name, baseline_value in baseline.items():
        current_value = current[metric_name]
        delta = current_value - baseline_value
        relative_delta = delta / baseline_value if baseline_value else 0.0
        metric_tolerance = tolerances[metric_name]

        if metric_name in LOWER_IS_WORSE:
            regressed = relative_delta < -metric_tolerance
        elif metric_name in HIGHER_IS_WORSE:
            regressed = relative_delta > metric_tolerance
        else:
            regressed = False

        report[metric_name] = {
            "baseline": baseline_value,
            "current": current_value,
            "relative_delta": relative_delta,
            "regressed": regressed,
        }
    return report


if __name__ == "__main__":
    from tests.conftest import load_ground_truth  # reuse the same label-parsing logic as tests

    positive_dir = Path(__file__).resolve().parent.parent / "data" / "images" / "positive"
    labeled_images = [(p, load_ground_truth(p)) for p in sorted(positive_dir.glob("*.jpg"))]

    detector = VehicleDetector()
    metrics = compute_current_metrics(detector, labeled_images)
    save_baseline(metrics)
    print(f"Baseline saved to {BASELINE_PATH}:")
    print(json.dumps(metrics, indent=2))

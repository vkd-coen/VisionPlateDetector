# Vehicle Detection - Automated Test Suite

An automated QA test suite that treats a pretrained YOLOv8 vehicle-detection model as a system
under test: functional correctness, accuracy metrics, robustness to real-world degradation,
latency/throughput SLAs, and regression tracking against a stored baseline - the workflow
described in Genetec's *Software Tester - Computer Vision & Machine Learning* posting. See
[PRD_Automated_Detection_Test_Suite.md](PRD_Automated_Detection_Test_Suite.md) for the full spec.

`<!-- once pushed to GitHub, replace the line below with: -->`
`![CI](https://github.com/<owner>/<repo>/actions/workflows/test.yml/badge.svg)`

## Test strategy

| Category | File | PRD | What it checks |
|---|---|---|---|
| Functional | `tests/test_functional.py` | 5.1 | Detects vehicles on positives, no false positives on negatives, IoU >= 0.5 vs. ground truth, correct class label |
| Accuracy | `tests/test_accuracy.py` | 5.2 | Precision/recall/F1 (scikit-learn), mAP@0.5 (torchmetrics), per-class breakdown |
| Robustness | `tests/test_robustness.py` | 5.3 | Detection confidence under 5 synthetic degradations: motion blur, low light, occlusion, rotation, JPEG compression |
| Performance | `tests/test_performance.py` | 5.4 | Average CPU inference latency SLA, FPS throughput on a sample video clip |
| Regression | `tests/test_regression.py` | 5.5 | Compares a fresh run against a stored baseline; includes a deliberately-degraded config to prove the comparison logic actually catches a real drop |

16 test cases total. All pass-fail thresholds in this suite are calibrated against an observed
YOLOv8n baseline on this dataset (2026-07-11) rather than picked arbitrarily - see each test
module's docstring and Known Limitations below for the reasoning and the real numbers.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/activate        # Windows
pip install -r requirements.txt
python -m src.data_loader       # pulls the COCO vehicle subset via fiftyone (one-time)
python -m src.video_utils       # synthesizes the sample clip used by the FPS test (one-time)
```

`data/images/`, `data/labels/`, `data/video/`, and `baselines/baseline_metrics.json` are already
committed to this repo, so a fresh clone can skip straight to **Run the suite** below - the two
commands above are only needed if you want to regenerate them (e.g. a larger sample, different
seed).

## Run the suite

```bash
pytest --html=reports/report.html --self-contained-html
```

> If `pytest` fails immediately with `ModuleNotFoundError: No module named 'lark'` (or similar)
> from a `launch_pytest`/ROS traceback, a global `PYTHONPATH` on your machine is injecting an
> unrelated system package's pytest plugin. Clear it for this command: `PYTHONPATH= pytest ...`
> (bash) or `$env:PYTHONPATH=''; pytest ...` (PowerShell). This is a local-machine quirk, not a
> project issue - GitHub Actions CI doesn't hit it.

To regenerate the regression baseline (e.g. after intentionally changing model config):

```bash
python -m src.regression
```

## Structure

```
data/
  images/positive/    50 curated COCO images containing car/truck/bus
  images/negative/    20 curated COCO images with no vehicles (false-positive check)
  labels/              ground-truth boxes per image, pixel xyxy (JSON)
  video/                synthesized short clip for the FPS throughput test
baselines/
  baseline_metrics.json   "known good" precision/recall/F1/mAP/latency, for regression comparison
src/
  model.py             YOLOv8 inference wrapper (car/truck/bus only)
  data_loader.py         pulls the COCO subset via fiftyone
  video_utils.py           synthesizes the sample clip from curated images
  metrics.py                IoU matching, precision/recall/F1, mAP@0.5
  augmentation.py             degraded-image variants (albumentations)
  regression.py                 baseline load/save + comparison logic
tests/
  conftest.py           shared fixtures (detector, image sets, ground truth)
  test_functional.py       PRD 5.1
  test_accuracy.py           PRD 5.2
  test_robustness.py           PRD 5.3
  test_performance.py            PRD 5.4
  test_regression.py               PRD 5.5
reports/                 generated HTML report
.github/workflows/test.yml   CI pipeline (GitHub Actions)
```

## Known limitations

- Dataset is a small curated COCO-2017 subset (50 positive / 20 negative images), not the full
  validation set - sized for fast iteration, not statistical rigor.
- Scoped to car/truck/bus only, per PRD.
- **YOLOv8n under-recalls `truck`** relative to `car`/`bus` on this dataset (observed baseline,
  2026-07-11: precision 0.67, recall 0.37, F1 0.48) - likely visual confusion with car/bus in the
  COCO ontology. The accuracy test suite's per-class floor is deliberately set below the aggregate
  floor to reflect this real, tracked baseline rather than hide it.
- **YOLOv8n is notably fragile to motion blur**: robust-detection rate (confidence >= 0.35 after
  degradation) drops to 26% under synthetic motion blur, vs. 80% under occlusion and JPEG
  compression, 74% under rotation, and 52% under low light. This is a genuine, documented model
  weakness surfaced by the robustness suite, not a test artifact - the kind of finding this
  project exists to catch. A production deployment on motion-heavy camera feeds would need either
  a larger model, motion-compensated preprocessing, or a lowered confidence threshold with
  downstream filtering.
- Functional tests use rate-based tolerances (not zero-tolerance) for miss rate and false-positive
  rate, calibrated against the observed YOLOv8n baseline - a nano model won't hit 100% on a raw,
  uncurated COCO sample, and COCO's own instance annotations aren't guaranteed exhaustive (an
  unlabeled background vehicle in a "negative" image is a labeling gap, not necessarily a model
  defect).
- CPU-only latency/FPS numbers are hardware-dependent; SLA constants in
  `tests/test_performance.py` are calibrated for the development machine and should be
  re-baselined on deployment/CI hardware.
- **Wall-clock latency is noisy on a shared dev machine**: repeated runs with identical model
  config and images showed swings up to +39% (70ms -> 98ms) purely from background system load,
  not a real slowdown. `src/regression.py`'s latency tolerance (60%) is set wide enough to absorb
  that noise and only flag a genuine multi-x regression; the absolute SLA ceiling in
  `tests/test_performance.py` is the authoritative latency check.
- The regression suite's baseline was generated from this same YOLOv8n configuration, so
  `test_current_run_matches_baseline` is expected to pass by construction; the deliberately
  degraded-config test (`test_regression_detection_catches_degraded_config`) is what actually
  proves the comparison logic detects a real drop.

## Resume bullet

Built an automated QA test suite (Python, pytest, YOLOv8/Ultralytics) that validates a
vehicle-detection model across 16 test cases spanning functional correctness, accuracy metrics
(precision/recall/F1/mAP@0.5 via scikit-learn and torchmetrics), robustness to synthetic
degradation (motion blur, low light, occlusion, rotation, JPEG compression via albumentations),
CPU latency/throughput SLAs, and regression tracking against a stored baseline - surfacing real
model weaknesses (e.g. a 26% robust-detection rate under motion blur) rather than hiding them
behind loosened thresholds, with CI (GitHub Actions) and an auto-generated HTML report.

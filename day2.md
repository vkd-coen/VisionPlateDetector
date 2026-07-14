# Day 2 Summary — Automated Detection Test Suite

Status: Day 2 complete (robustness, performance, regression tests, and CI — PRD sections 5.3,
5.4, 5.5). All 16 tests pass end-to-end against real, calibrated thresholds (not artificially
loosened to force a green run).

## Most important files

**`src/augmentation.py`** — `VARIANTS` dict of 5 albumentations-based degradations: motion blur,
low light (brightness/contrast reduction), occlusion (`CoarseDropout` patch), rotation, JPEG
compression. `apply_variant(image, name, seed)` / `generate_variants(image, seed)` are
deterministic (wrapped in a seeded `A.Compose`) for reproducible tests.

**`tests/test_robustness.py`** — PRD 5.3: for each of the 50 positive images, applies each
degradation, reruns detection, and checks the fraction of images that stay at or above a
confidence floor (`MIN_CONFIDENCE_FLOOR=0.35`, deliberately above the detector's own conf=0.25
cutoff). Per-variant `MIN_ROBUST_RATE` floors are calibrated from the real observed baseline
(see Findings below), not guessed.

**`src/video_utils.py`** + **`data/video/sample_clip.mp4`** — synthesizes a 10s/50-frame clip
by stitching the curated positive images together (`cv2.VideoWriter`, 5fps, 640x480). No external
video asset needed; fully reproducible via `python -m src.video_utils`.

**`tests/test_performance.py`** — PRD 5.4: `test_average_inference_latency` times CPU inference
per image (with a warm-up call excluded) and asserts under `MAX_AVG_LATENCY_MS=400`;
`test_video_clip_fps` reads `sample_clip.mp4` frame-by-frame and asserts throughput above
`MIN_FPS=2.0`.

**`src/regression.py`** — baseline storage/comparison engine:
- `compute_current_metrics(detector, labeled_images)` → precision/recall/F1/mAP@0.5/avg latency.
- `save_baseline` / `load_baseline` — JSON round-trip to `baselines/baseline_metrics.json`.
- `compare_to_baseline(current, baseline, tolerance)` — per-metric regression flagging.
  `DEFAULT_TOLERANCE` is **per-metric**: 0.02 (2%) for the deterministic accuracy metrics
  (precision/recall/F1/mAP50), 0.60 (60%) for `avg_latency_ms` specifically, because wall-clock
  CPU timing on this dev machine swung up to +39% between otherwise-identical runs purely from
  background system load — see Findings below.
- Run directly (`python -m src.regression`) to regenerate the baseline from the current detector
  config.

**`baselines/baseline_metrics.json`** — the stored "known good" reference:
precision 0.695, recall 0.549, F1 0.613, mAP@0.5 0.496, avg latency ~70ms.

**`tests/test_regression.py`** — PRD 5.5:
- `test_current_run_matches_baseline` — same config as baseline, expect no flagged regression.
- `test_regression_detection_catches_degraded_config` — **deliberately** reruns with confidence
  threshold raised 0.25→0.75 (starves recall) and asserts the comparison logic *does* flag a
  regression. This is the PRD's required proof that the regression-detection logic actually works,
  not just that it stays quiet when nothing changed.

**`.github/workflows/test.yml`** — GitHub Actions CI: checkout → setup Python 3.10 (pip cache) →
install `libgl1`/`libglib2.0-0` (opencv-python needs these on headless Ubuntu runners) → `pip
install -r requirements.txt` → `pytest --html=reports/report.html --self-contained-html` → upload
the HTML report as a build artifact. **Not yet verified live** — requires pushing to a GitHub
remote (tomorrow's task), which is a manual step per the PRD's own scope note.

**`README.md`** — now includes a full test-strategy table (category → file → PRD section → what
it checks), setup/run/reproduce commands, structure tree, known limitations, and a drafted resume
bullet (PRD deliverable #4).

## Commands to run the project

```bash
# one-time setup (already done in this repo - data/images, data/labels, data/video, and
# baselines/baseline_metrics.json are committed, so a fresh clone can skip straight to "run")
python -m venv .venv
./.venv/Scripts/activate            # Windows
pip install -r requirements.txt
python -m src.data_loader           # pulls dataset into data/images/, data/labels/
python -m src.video_utils           # synthesizes data/video/sample_clip.mp4

# run the full suite (16 tests)
pytest --html=reports/report.html --self-contained-html

# regenerate the regression baseline (e.g. after an intentional model/config change)
python -m src.regression
```

On this machine, a global `PYTHONPATH` (ROS2) breaks pytest plugin autoload — same workaround as
Day 1:
```bash
PYTHONPATH= pytest --html=reports/report.html --self-contained-html   # bash
$env:PYTHONPATH=''; pytest --html=reports/report.html --self-contained-html   # PowerShell
```
(This is local-machine-specific; GitHub Actions CI doesn't hit it.)

## Where results are

- **Test report:** `reports/report.html` (16 tests, self-contained HTML).
- **Regression baseline:** `baselines/baseline_metrics.json`.
- **Sample video:** `data/video/sample_clip.mp4`.
- **CI config:** `.github/workflows/test.yml` (runs on push/PR once the repo is on GitHub).

## Real findings surfaced (documented, not hidden behind loosened thresholds)

1. **YOLOv8n is notably fragile to motion blur.** Robust-detection rate (confidence ≥ 0.35 after
   degradation) on the 50 positive images:
   - motion_blur: **26%**
   - low_light: 52%
   - rotation: 74%
   - occlusion: 80%
   - jpeg_compression: 80%

   Motion blur is the clear outlier — a genuine model weakness, not a test bug. Documented in
   README's Known Limitations with a note that a production deployment on motion-heavy camera
   feeds would need a larger model, motion-compensated preprocessing, or confidence-threshold
   tuning with downstream filtering.

2. **Wall-clock CPU latency is noisy on this dev machine.** Three back-to-back runs of the exact
   same detector config against the exact same images measured 70ms, 72ms, and 98ms average
   latency — a +39% swing from background system load alone, unrelated to any real regression.
   This forced moving from a single scalar regression tolerance to a per-metric tolerance dict
   (`src/regression.py::DEFAULT_TOLERANCE`): tight (2%) for the deterministic accuracy metrics,
   wide (60%) for latency specifically. The absolute SLA ceiling in `test_performance.py`
   (400ms) remains the authoritative correctness check for latency; the regression comparison
   only guards against a dramatic (multi-x) slowdown.

3. Carried forward from Day 1: **`truck` recall (0.37)** is now the tracked regression baseline
   going forward — any further drop beyond the per-class floor will fail the accuracy suite.

## Issues hit and fixed along the way

1. **Ultralytics numpy-array inference** — `VehicleDetector.predict` originally only accepted a
   file path; robustness/regression tests need to run inference on in-memory augmented arrays.
   Widened the signature to `str | Path | np.ndarray` in `src/model.py`.
2. **albumentations reproducibility** — passing `random_state=` directly to a transform's
   `__call__` silently no-ops (no error, but non-deterministic output). Fixed by wrapping each
   transform in a seeded `A.Compose([...], seed=seed)`, which *is* deterministic.
3. **Regression-tolerance flakiness** (see Finding #2 above) — required redesigning
   `compare_to_baseline` to accept per-metric tolerances instead of one scalar for every metric.

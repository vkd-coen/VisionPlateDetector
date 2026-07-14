# PRD: Automated Test Suite for a Vehicle & License Plate Detection Model

**Author:** Vik
**Purpose:** Portfolio project targeting the *Software Tester – Computer Vision & Machine Learning* role at Genetec
**Timeline:** 2 days
**Language:** Python

---

## 1. Problem Statement

CV/ML models are notoriously hard to validate with traditional QA methods — "does it work" isn't a binary pass/fail, it's a distribution of accuracy across conditions (lighting, angle, occlusion, blur). This project builds an automated test framework that treats a vehicle/plate detection model as a system under test, applying real QA discipline: defined test cases, measurable pass/fail criteria, regression tracking, and a reportable output — exactly the workflow described in the Genetec posting (functional/exploratory/automated testing, test strategy design, performance analysis, defect documentation).

## 2. Goals

- Demonstrate the ability to design and execute a structured test strategy for an ML/CV system.
- Produce quantifiable, reproducible test results (not just "it looks right").
- Show automation skill (pytest + CI) and reporting/documentation discipline.
- Stress-test the model against edge cases relevant to smart-camera deployments (blur, low light, occlusion, angle).
- End with a clean, demoable GitHub repo + report that can be linked directly from the resume.

## 3. Non-Goals

- Not building or training a custom detection model — a pretrained YOLOv8 model is the system under test.
- Not building a full CI/CD pipeline — a single GitHub Actions workflow that runs the suite is enough.
- Not covering multi-camera/live video streaming — static image + short video clip testing only.

## 4. System Under Test (SUT)

- **Model:** YOLOv8n or YOLOv8s (Ultralytics, pretrained on COCO) for vehicle detection.
  - Stretch goal if time allows: swap in an open-source plate-detection model for a second SUT, to show testing isn't tied to one model.
- **Input data:** Public vehicle image dataset (subset of COCO validation images filtered to vehicle classes) + a small self-collected/augmented set of edge-case images.

## 5. Test Strategy

### 5.1 Functional Tests
- Detects at least one vehicle in known-positive images.
- Returns zero detections on known-negative images (no vehicles present) — false positive check.
- Bounding box IoU against ground truth exceeds a defined threshold (e.g. 0.5) on a labeled sample.
- Correct class label assigned (car/truck/bus) where ground truth is available.

### 5.2 Accuracy / Performance Metrics
- Precision, Recall, F1 (via scikit-learn) over the labeled test set.
- mAP@0.5 computed over the same set.
- Per-class breakdown (car vs. truck vs. bus) to catch class-specific weaknesses.

### 5.3 Robustness / Edge-Case Tests
Programmatically generate degraded variants of the same base images using `albumentations`:
- Motion blur
- Low light / underexposure
- Partial occlusion (synthetic patch overlay)
- Rotation / off-angle
- JPEG compression artifacts

Each variant re-run through the model; test asserts detection confidence doesn't fall below a defined minimum threshold, flagging robustness regressions.

### 5.4 Latency / Throughput Tests
- Measure average inference time per image (CPU) — assert under a defined SLA (e.g. <200ms/image).
- FPS estimate on a short video clip.

### 5.5 Regression Tests
- Store baseline metrics (precision/recall/mAP/latency) from a "known good" model version.
- Test suite compares new run against baseline, flags any metric drop beyond a tolerance (e.g. >2%).
- Simulates the "test new hardware/software features without breaking existing behavior" requirement.

## 6. Tooling & Stack

| Layer | Tool |
|---|---|
| Detection model | YOLOv8 (Ultralytics) |
| Image processing | OpenCV |
| Test runner | pytest |
| Test reporting | pytest-html |
| Metrics | scikit-learn, manual mAP calc |
| Robustness augmentation | albumentations |
| Dataset | COCO subset (vehicle classes) |
| CI | GitHub Actions |
| Version control | Git / GitHub |

> **Note on GitHub workflow:** repo creation, commits, and CI file setup are done manually — either via standard `git` CLI (PAT stored as an environment variable / credential manager) or by scripting direct calls to the GitHub REST API (e.g. `PUT /repos/{owner}/{repo}/contents/{path}`) with `Authorization: Bearer <PAT>`. No agentic/MCP tooling is used for GitHub interaction; all commits and pushes are initiated and reviewed manually.

## 7. Deliverables

1. **GitHub repo** with clear structure:
   ```
   /data          → sample test images + labels
   /tests         → pytest test modules (functional, robustness, performance, regression)
   /reports       → generated HTML/metrics reports
   /src           → helper modules (metrics calc, augmentation, model wrapper)
   README.md      → setup, how to run, sample report screenshot
   .github/workflows/test.yml → CI pipeline
   ```
2. **HTML test report** (auto-generated) summarizing pass/fail counts, metrics, and flagged regressions.
3. **README** written like a real QA deliverable: test strategy summary, how to reproduce, known limitations.
4. **One-paragraph resume bullet** distilled from the finished project (drafted at the end).

## 8. Success Criteria

- Test suite runs end-to-end via a single command (`pytest --html=report.html`).
- CI badge shows passing build on GitHub.
- At least 15–20 discrete test cases across the 5 categories above.
- Report clearly shows at least one deliberately-introduced "regression" (e.g. testing YOLOv8n vs YOLOv8s) to prove the regression-detection logic actually works.

## 9. 2-Day Plan

**Day 1**
- Set up repo, environment, dependencies.
- Load YOLOv8 pretrained model, wrap in a simple inference helper.
- Pull/prepare COCO vehicle subset + ground truth labels.
- Write functional tests (5.1) and accuracy/metrics tests (5.2).

**Day 2**
- Build robustness augmentation pipeline + edge-case tests (5.3).
- Add latency/performance tests (5.4).
- Add regression baseline + comparison logic (5.5).
- Set up GitHub Actions CI.
- Generate final HTML report, write README, polish repo for portfolio use.

## 10. Risks

- **COCO vehicle subset may be large** → mitigate by using a small curated 50–100 image sample rather than full dataset.
- **mAP calculation complexity** → use a lightweight existing implementation (e.g. `torchmetrics` MeanAveragePrecision) rather than writing mAP from scratch to save time.
- **Two-day scope creep** → treat sections 5.3–5.5 as "if time allows" after 5.1–5.2 are solid; a smaller, fully-working suite beats a larger, half-broken one.

---

**Next steps:** Confirm this scope, then determine the right Claude model/workflow (Claude Code vs. chat-driven) and any MCP/plugins needed (GitHub, filesystem, etc.) before starting implementation.

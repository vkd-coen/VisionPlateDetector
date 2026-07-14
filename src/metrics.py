"""Detection metrics: IoU matching, precision/recall/F1 (scikit-learn), mAP@0.5 (torchmetrics)."""
from typing import NamedTuple

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torchmetrics.detection import MeanAveragePrecision

from src.model import Detection


class GroundTruth(NamedTuple):
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    class_name: str


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


def match_detections_by_iou(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Greedy best-IoU matching, ignoring class labels. Returns (det_index, gt_index) pairs."""
    gt_matched = [False] * len(ground_truths)
    det_matched = [False] * len(detections)

    pairs = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(ground_truths):
            iou = compute_iou(det.box, gt.box)
            if iou >= iou_threshold:
                pairs.append((iou, di, gi))
    pairs.sort(key=lambda p: p[0], reverse=True)

    matches = []
    for _, di, gi in pairs:
        if not det_matched[di] and not gt_matched[gi]:
            det_matched[di] = True
            gt_matched[gi] = True
            matches.append((di, gi))

    return matches


def match_detections(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> tuple[list[bool], list[bool]]:
    """Greedy best-IoU matching. Returns (gt_matched, det_matched) boolean flags,
    where a match additionally requires the predicted class to equal the ground-truth class."""
    gt_matched = [False] * len(ground_truths)
    det_matched = [False] * len(detections)

    pairs = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(ground_truths):
            iou = compute_iou(det.box, gt.box)
            if iou >= iou_threshold and det.class_name == gt.class_name:
                pairs.append((iou, di, gi))
    pairs.sort(key=lambda p: p[0], reverse=True)

    for _, di, gi in pairs:
        if not det_matched[di] and not gt_matched[gi]:
            det_matched[di] = True
            gt_matched[gi] = True

    return gt_matched, det_matched


def precision_recall_f1(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision/recall/F1 over a single image's detections vs. ground truth, via scikit-learn.

    Built as binary y_true/y_pred arrays: each ground truth contributes a 1 (matched -> also
    predicted 1, unmatched -> predicted 0/FN); each unmatched detection contributes a FP (true 0,
    predicted 1).
    """
    gt_matched, det_matched = match_detections(detections, ground_truths, iou_threshold)

    y_true = [1] * len(ground_truths) + [0] * det_matched.count(False)
    y_pred = [1 if m else 0 for m in gt_matched] + [1] * det_matched.count(False)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def aggregate_prf1(
    per_image: list[tuple[list[Detection], list[GroundTruth]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Micro-averaged precision/recall/F1 across a dataset (sums TP/FP/FN across images first)."""
    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for detections, ground_truths in per_image:
        gt_matched, det_matched = match_detections(detections, ground_truths, iou_threshold)
        y_true_all += [1] * len(ground_truths) + [0] * det_matched.count(False)
        y_pred_all += [1 if m else 0 for m in gt_matched] + [1] * det_matched.count(False)

    return {
        "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
        "recall": recall_score(y_true_all, y_pred_all, zero_division=0),
        "f1": f1_score(y_true_all, y_pred_all, zero_division=0),
    }


def per_class_prf1(
    per_image: list[tuple[list[Detection], list[GroundTruth]]],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    breakdown = {}
    for class_name in class_names:
        filtered = [
            (
                [d for d in dets if d.class_name == class_name],
                [g for g in gts if g.class_name == class_name],
            )
            for dets, gts in per_image
        ]
        breakdown[class_name] = aggregate_prf1(filtered, iou_threshold)
    return breakdown


def compute_map50(
    per_image: list[tuple[list[Detection], list[GroundTruth]]],
    class_name_to_id: dict[str, int],
) -> float:
    """mAP@0.5 across a dataset via torchmetrics."""
    metric = MeanAveragePrecision(iou_thresholds=[0.5])

    preds, targets = [], []
    for detections, ground_truths in per_image:
        preds.append(
            {
                "boxes": torch.tensor([d.box for d in detections]) if detections else torch.empty((0, 4)),
                "scores": torch.tensor([d.confidence for d in detections]) if detections else torch.empty((0,)),
                "labels": torch.tensor([class_name_to_id[d.class_name] for d in detections], dtype=torch.long)
                if detections
                else torch.empty((0,), dtype=torch.long),
            }
        )
        targets.append(
            {
                "boxes": torch.tensor([g.box for g in ground_truths]) if ground_truths else torch.empty((0, 4)),
                "labels": torch.tensor([class_name_to_id[g.class_name] for g in ground_truths], dtype=torch.long)
                if ground_truths
                else torch.empty((0,), dtype=torch.long),
            }
        )

    metric.update(preds, targets)
    result = metric.compute()
    return float(result["map_50"])

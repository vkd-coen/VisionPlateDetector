"""PRD 5.3 - generates degraded variants of an image to stress-test detection robustness.

`severity` (default 1.0) scales each transform's intensity - 1.0 reproduces the exact parameters
this module always used, so existing callers (e.g. tests/test_robustness.py, which never passes
severity) are unaffected. It exists for the Streamlit demo's severity slider.
"""
import albumentations as A
import cv2
import numpy as np

VARIANT_NAMES = ["motion_blur", "low_light", "occlusion", "rotation", "jpeg_compression"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _odd(value: float) -> int:
    n = max(3, round(value))
    return n if n % 2 == 1 else n + 1


def _build_transform(variant_name: str, severity: float = 1.0) -> A.BasicTransform:
    if variant_name == "motion_blur":
        return A.MotionBlur(blur_limit=(_odd(15 * severity), _odd(25 * severity)), p=1.0)
    if variant_name == "low_light":
        return A.RandomBrightnessContrast(
            brightness_limit=(
                _clamp(-0.6 * severity, -1.0, 0.0),
                _clamp(-0.5 * severity, -1.0, 0.0),
            ),
            contrast_limit=(-0.2, -0.1),
            p=1.0,
        )
    if variant_name == "occlusion":
        hole_range = (_clamp(0.3 * severity, 0.05, 0.9), _clamp(0.4 * severity, 0.05, 0.9))
        return A.CoarseDropout(
            num_holes_range=(1, 1), hole_height_range=hole_range, hole_width_range=hole_range,
            fill=0, p=1.0,
        )
    if variant_name == "rotation":
        return A.Rotate(limit=(15 * severity, 25 * severity), border_mode=cv2.BORDER_REPLICATE, p=1.0)
    if variant_name == "jpeg_compression":
        return A.ImageCompression(
            quality_range=(
                int(_clamp(round(10 / severity), 1, 100)),
                int(_clamp(round(20 / severity), 1, 100)),
            ),
            p=1.0,
        )
    raise KeyError(f"unknown variant: {variant_name}")


VARIANTS: dict[str, A.BasicTransform] = {name: _build_transform(name) for name in VARIANT_NAMES}


def apply_variant(
    image: np.ndarray,
    variant_name: str,
    seed: int = 0,
    severity: float = 1.0,
    boxes: list[tuple[float, float, float, float]] | None = None,
):
    """If `boxes` (pixel xyxy) is given, they're transformed alongside the image and returned as
    `(image, boxes)` - required for `rotation`, which actually moves pixels (so a ground-truth box
    in the *original* frame would otherwise no longer line up with the object in the rotated
    image); harmless no-op for the other variants, which don't move anything spatially. Without
    `boxes`, returns just the image, as before."""
    transform = VARIANTS[variant_name] if severity == 1.0 else _build_transform(variant_name, severity)

    if boxes is None:
        pipeline = A.Compose([transform], seed=seed)
        return pipeline(image=image)["image"]

    pipeline = A.Compose(
        [transform], seed=seed,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], clip=True),
    )
    result = pipeline(image=image, bboxes=boxes, labels=list(range(len(boxes))))
    return result["image"], [tuple(box) for box in result["bboxes"]]


def generate_variants(image: np.ndarray, seed: int = 0, severity: float = 1.0) -> dict[str, np.ndarray]:
    return {name: apply_variant(image, name, seed=seed, severity=severity) for name in VARIANTS}

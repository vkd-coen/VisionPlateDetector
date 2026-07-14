"""PRD 5.3 - generates degraded variants of an image to stress-test detection robustness."""
import albumentations as A
import cv2
import numpy as np

VARIANTS: dict[str, A.BasicTransform] = {
    "motion_blur": A.MotionBlur(blur_limit=(15, 25), p=1.0),
    "low_light": A.RandomBrightnessContrast(
        brightness_limit=(-0.6, -0.5), contrast_limit=(-0.2, -0.1), p=1.0
    ),
    "occlusion": A.CoarseDropout(
        num_holes_range=(1, 1), hole_height_range=(0.3, 0.4), hole_width_range=(0.3, 0.4),
        fill=0, p=1.0,
    ),
    "rotation": A.Rotate(limit=(15, 25), border_mode=cv2.BORDER_REPLICATE, p=1.0),
    "jpeg_compression": A.ImageCompression(quality_range=(10, 20), p=1.0),
}


def apply_variant(image: np.ndarray, variant_name: str, seed: int = 0) -> np.ndarray:
    pipeline = A.Compose([VARIANTS[variant_name]], seed=seed)
    return pipeline(image=image)["image"]


def generate_variants(image: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    return {name: apply_variant(image, name, seed=seed) for name in VARIANTS}

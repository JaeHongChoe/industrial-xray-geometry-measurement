"""X-ray 3-pin detector package."""

from __future__ import annotations

from .circles import Circle, compute_eccentricity, detect_circles
from .detection import detect_3pins, detect_jig_cross
from .pipeline import infer_image
from .preprocess import load_grayscale, preprocess_image
from .products import classify_group
from .vhole_v7 import V7ONNXDetector, VHoleResult

__all__ = [
    "Circle",
    "V7ONNXDetector",
    "VHoleResult",
    "classify_group",
    "compute_eccentricity",
    "detect_3pins",
    "detect_circles",
    "detect_jig_cross",
    "infer_image",
    "load_grayscale",
    "preprocess_image",
]

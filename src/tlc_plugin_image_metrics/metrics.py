# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image quality metrics for ML training data analysis.

Each metric is a pure function that takes a PIL Image (RGB) and returns a float.
Metrics are grouped into categories for UI presentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class MetricDef:
    """Definition of a single image quality metric."""

    id: str
    name: str
    description: str
    category: str
    icon: str
    unit: str = ""


# ---------------------------------------------------------------------------
# Metric definitions — order matters for UI
# ---------------------------------------------------------------------------

METRICS: list[MetricDef] = [
    # ── Brightness & Exposure ──
    MetricDef(
        id="brightness",
        name="Brightness",
        description="Mean luminance (0-255). Detects over/underexposed images that can confuse models.",
        category="Brightness & Exposure",
        icon="☀️",
    ),
    MetricDef(
        id="overexposure",
        name="Overexposure",
        description="Fraction of near-white pixels (>240). High values indicate washed-out images with lost detail.",
        category="Brightness & Exposure",
        icon="🔆",
        unit="%",
    ),
    MetricDef(
        id="underexposure",
        name="Underexposure",
        description="Fraction of near-black pixels (<15). High values indicate dark images where features are hidden.",
        category="Brightness & Exposure",
        icon="🌑",
        unit="%",
    ),
    MetricDef(
        id="dynamic_range",
        name="Dynamic Range",
        description="Difference between 95th and 5th percentile intensity. Low range means flat, low-contrast images.",
        category="Brightness & Exposure",
        icon="▤",
    ),
    # ── Sharpness & Noise ──
    MetricDef(
        id="sharpness",
        name="Sharpness",
        description="Laplacian variance — measures edge strength. "
        "Low values indicate blurry images that hurt model performance.",
        category="Sharpness & Noise",
        icon="◇",
    ),
    MetricDef(
        id="noise_estimate",
        name="Noise Estimate",
        description="Estimated sensor noise via median absolute deviation. High noise degrades feature extraction.",
        category="Sharpness & Noise",
        icon="📡",
    ),
    MetricDef(
        id="edge_density",
        name="Edge Density",
        description="Fraction of pixels that are edges (Canny). "
        "Useful for detecting overly simple or overly complex scenes.",
        category="Sharpness & Noise",
        icon="✏️",
        unit="%",
    ),
    # ── Color ──
    MetricDef(
        id="contrast",
        name="Contrast",
        description="Standard deviation of luminance. Low contrast images lack visual discriminability.",
        category="Color",
        icon="🎚️",
    ),
    MetricDef(
        id="saturation",
        name="Saturation",
        description="Mean HSV saturation (0-255). Low saturation often means grayscale or washed-out color.",
        category="Color",
        icon="🎨",
    ),
    MetricDef(
        id="colorfulness",
        name="Colorfulness",
        description="Hasler & Süsstrunk colorfulness metric. Quantifies how vivid and varied the colors are.",
        category="Color",
        icon="🌈",
    ),
    # ── Information ──
    MetricDef(
        id="entropy",
        name="Entropy",
        description="Shannon entropy of the grayscale histogram. Higher entropy = more information content.",
        category="Information",
        icon="🧮",
        unit="bits",
    ),
    # ── Geometry ──
    MetricDef(
        id="width",
        name="Width",
        description="Image width in pixels. Useful for finding images that differ from the expected resolution.",
        category="Geometry",
        icon="↔️",
        unit="px",
    ),
    MetricDef(
        id="height",
        name="Height",
        description="Image height in pixels. Useful for finding images that differ from the expected resolution.",
        category="Geometry",
        icon="↕️",
        unit="px",
    ),
    MetricDef(
        id="aspect_ratio",
        name="Aspect Ratio",
        description="Width / height. Extreme ratios may need special padding/cropping during training.",
        category="Geometry",
        icon="📐",
    ),
    MetricDef(
        id="area",
        name="Area",
        description="Total pixel count (width × height). Useful for finding unusually small or large images.",
        category="Geometry",
        icon="📏",
        unit="px",
    ),
]

METRIC_BY_ID: dict[str, MetricDef] = {m.id: m for m in METRICS}
ALL_METRIC_IDS: list[str] = [m.id for m in METRICS]
CATEGORIES: list[str] = list(dict.fromkeys(m.category for m in METRICS))


# ---------------------------------------------------------------------------
# Compute functions — one per metric, all take PIL Image (RGB) + numpy array
# ---------------------------------------------------------------------------


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to grayscale float64."""
    gray: np.ndarray = np.dot(rgb[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
    return gray


def compute_brightness(gray: np.ndarray, **_: Any) -> float:
    """Mean luminance 0-255."""
    return float(np.mean(gray))


def compute_overexposure(gray: np.ndarray, **_: Any) -> float:
    """Fraction of pixels > 240 (percent)."""
    return float(np.mean(gray > 240) * 100)


def compute_underexposure(gray: np.ndarray, **_: Any) -> float:
    """Fraction of pixels < 15 (percent)."""
    return float(np.mean(gray < 15) * 100)


def compute_dynamic_range(gray: np.ndarray, **_: Any) -> float:
    """95th - 5th percentile of grayscale intensity."""
    p5, p95 = np.percentile(gray, [5, 95])
    return float(p95 - p5)


def compute_sharpness(gray: np.ndarray, **_: Any) -> float:
    """Laplacian variance — higher = sharper.

    Uses a 3x3 Laplacian kernel convolved via numpy (no scipy dependency).
    """
    # Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    # Pad with edge values, then convolve manually
    padded = np.pad(gray, 1, mode="edge")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * padded[1:-1, 1:-1]
    return float(np.var(lap))


def compute_noise_estimate(gray: np.ndarray, **_: Any) -> float:
    """Noise estimate via median absolute deviation of Laplacian.

    Uses the same 3x3 Laplacian kernel as sharpness.
    """
    padded = np.pad(gray, 1, mode="edge")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * padded[1:-1, 1:-1]
    # Robust noise estimate: sigma ≈ MAD / 0.6745
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return mad / 0.6745 if mad > 0 else 0.0


def compute_edge_density(img: Image.Image, **_: Any) -> float:
    """Fraction of edge pixels (percent) using PIL FIND_EDGES."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges)
    threshold = 30
    return float(np.mean(arr > threshold) * 100)


def compute_contrast(gray: np.ndarray, **_: Any) -> float:
    """Standard deviation of grayscale intensity."""
    return float(np.std(gray))


def compute_saturation(rgb: np.ndarray, **_: Any) -> float:
    """Mean saturation in HSV space (0-255)."""
    # Fast vectorized RGB→S conversion
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    # S = delta / cmax where cmax > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(cmax > 0, delta / cmax, 0.0)
    return float(np.mean(sat) * 255)


def compute_colorfulness(rgb: np.ndarray, **_: Any) -> float:
    """Hasler & Süsstrunk colorfulness metric."""
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))
    std_root = math.sqrt(std_rg**2 + std_yb**2)
    mean_root = math.sqrt(mean_rg**2 + mean_yb**2)
    return std_root + 0.3 * mean_root


def compute_entropy(gray: np.ndarray, **_: Any) -> float:
    """Shannon entropy (bits) of the grayscale histogram."""
    hist, _edges = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_width(img: Image.Image, **_: Any) -> float:
    """Image width in pixels."""
    return float(img.size[0])


def compute_height(img: Image.Image, **_: Any) -> float:
    """Image height in pixels."""
    return float(img.size[1])


def compute_aspect_ratio(img: Image.Image, **_: Any) -> float:
    """Width / height."""
    w, h = img.size
    return round(w / h, 4) if h > 0 else 0.0


def compute_area(img: Image.Image, **_: Any) -> float:
    """Total pixel count."""
    w, h = img.size
    return float(w * h)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_COMPUTE_FNS: dict[str, Any] = {
    "brightness": compute_brightness,
    "overexposure": compute_overexposure,
    "underexposure": compute_underexposure,
    "dynamic_range": compute_dynamic_range,
    "sharpness": compute_sharpness,
    "noise_estimate": compute_noise_estimate,
    "edge_density": compute_edge_density,
    "contrast": compute_contrast,
    "saturation": compute_saturation,
    "colorfulness": compute_colorfulness,
    "entropy": compute_entropy,
    "width": compute_width,
    "height": compute_height,
    "aspect_ratio": compute_aspect_ratio,
    "area": compute_area,
}


def compute_metrics_for_image(
    img: Image.Image,
    metric_ids: list[str],
) -> dict[str, float]:
    """Compute requested metrics for a single PIL Image.

    Converts to numpy once and reuses arrays across all metrics.

    Args:
        img: PIL Image in RGB mode.
        metric_ids: List of metric IDs to compute.

    Returns:
        Dict mapping metric_id → float value.

    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.array(img)
    gray = _to_gray(rgb)

    results: dict[str, float] = {}
    for mid in metric_ids:
        fn = _COMPUTE_FNS.get(mid)
        if fn is None:
            continue
        try:
            results[mid] = fn(img=img, gray=gray, rgb=rgb)
        except Exception:
            results[mid] = float("nan")
    return results

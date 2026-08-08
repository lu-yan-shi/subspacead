"""Object localization for industrial inspection images.

Uses DINOv2 attention-based saliency masks (already computed during feature extraction)
as the primary localization signal, with OpenCV contour detection as a fallback.

Provides:
    - Multiple localization strategies (saliency, contour, manual, auto)
    - ROI cropping with configurable margin
    - Coordinate mapping from ROI space back to original image space
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================
# Data Classes
# ============================================================


@dataclass
class LocalizationResult:
    """Result of object localization."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original image coordinates
    mask: Optional[np.ndarray] = None  # binary foreground mask (H, W), same size as image
    confidence: float = 0.0
    method: str = "none"

    @property
    def area(self) -> int:
        x, y, w, h = self.bbox
        return w * h

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def is_valid(self) -> bool:
        x, y, w, h = self.bbox
        return w > 0 and h > 0


# ============================================================
# Localization Strategies
# ============================================================


def localize_saliency(
    saliency_map: np.ndarray,
    image_shape: Tuple[int, int],
    threshold_method: str = "otsu",
    percentile: float = 0.70,
    min_area_ratio: float = 0.05,
    kernel_size: int = 7,
) -> LocalizationResult:
    """Locate the foreground object using a DINOv2 saliency/attention map.

    The saliency map is a (h_p, w_p) array of attention scores from DINOv2.
    It is first resized to the original image resolution, then thresholded
    to produce a binary foreground mask, from which the largest connected
    component's bounding box is extracted.

    Args:
        saliency_map: DINOv2 attention map (h_p, w_p), values in arbitrary range
        image_shape: Original image dimensions as (H, W)
        threshold_method: "otsu" or "percentile"
        percentile: Percentile threshold (only for "percentile" method)
        min_area_ratio: Minimum foreground area as fraction of image area
        kernel_size: Morphological kernel size for cleaning

    Returns:
        LocalizationResult with bbox, mask, confidence
    """
    img_h, img_w = image_shape
    min_area = int(img_w * img_h * min_area_ratio)

    # Resize saliency map to original image size
    if saliency_map.shape[:2] != (img_h, img_w):
        saliency_resized = cv2.resize(
            saliency_map.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        saliency_resized = saliency_map.astype(np.float32)

    # Normalize to 0-255
    saliency_norm = cv2.normalize(saliency_resized, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold
    if threshold_method == "otsu":
        _, binary = cv2.threshold(
            saliency_norm.astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    elif threshold_method == "percentile":
        thresh_val = np.percentile(saliency_norm, percentile * 100)
        _, binary = cv2.threshold(
            saliency_norm.astype(np.uint8), thresh_val, 255, cv2.THRESH_BINARY,
        )
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    # Morphological cleanup
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find largest connected component
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("Saliency localization: no contours found, falling back to full image")
        return LocalizationResult(
            bbox=(0, 0, img_w, img_h),
            mask=None,
            confidence=0.0,
            method="saliency_fallback",
        )

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        logger.warning(
            "Saliency localization: largest contour area %.0f < min %.0f, falling back",
            area, min_area,
        )
        return LocalizationResult(
            bbox=(0, 0, img_w, img_h),
            mask=None,
            confidence=area / (img_w * img_h),
            method="saliency_fallback",
        )

    x, y, w, h = cv2.boundingRect(largest)

    # Compute confidence: ratio of contour area to bbox area (fill ratio)
    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0
    coverage = area / (img_w * img_h)
    confidence = float(0.5 * fill_ratio + 0.5 * min(coverage / 0.5, 1.0))

    return LocalizationResult(
        bbox=(x, y, w, h),
        mask=(binary > 0).astype(np.uint8),
        confidence=confidence,
        method="saliency",
    )


def localize_contour(
    image_np: np.ndarray,
    edge_low: int = 50,
    edge_high: int = 150,
    dilate_iterations: int = 3,
    min_area_ratio: float = 0.05,
) -> LocalizationResult:
    """Locate object using Canny edge detection + largest external contour.

    This is the fallback strategy when DINOv2 saliency is unavailable or
    produces degenerate results.

    Args:
        image_np: BGR or grayscale image as numpy array (H, W) or (H, W, C)
        edge_low: Canny low threshold
        edge_high: Canny high threshold
        dilate_iterations: Dilation iterations to close edge gaps
        min_area_ratio: Minimum contour area as fraction of image area

    Returns:
        LocalizationResult
    """
    img_h, img_w = image_np.shape[:2]
    min_area = int(img_w * img_h * min_area_ratio)

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np

    # Edge detection
    edges = cv2.Canny(gray, edge_low, edge_high)

    # Dilate to close gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.warning("Contour localization: no edges found")
        return LocalizationResult(
            bbox=(0, 0, img_w, img_h),
            confidence=0.0,
            method="contour_fallback",
        )

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        logger.warning("Contour localization: area too small, using full image")
        return LocalizationResult(
            bbox=(0, 0, img_w, img_h),
            confidence=area / (img_w * img_h),
            method="contour_fallback",
        )

    x, y, w, h = cv2.boundingRect(largest)
    confidence = min(area / (img_w * img_h * 0.3), 1.0)

    return LocalizationResult(
        bbox=(x, y, w, h),
        confidence=float(confidence),
        method="contour",
    )


def localize_manual(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> LocalizationResult:
    """Create a localization result from a user-provided bounding box.

    Args:
        bbox: (x, y, w, h) in image coordinates
        image_shape: (H, W) of the original image

    Returns:
        LocalizationResult
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape

    # Clamp to image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return LocalizationResult(
            bbox=(0, 0, img_w, img_h),
            confidence=0.0,
            method="manual_invalid",
        )

    return LocalizationResult(
        bbox=(x, y, w, h),
        confidence=1.0,
        method="manual",
    )


# ============================================================
# ROI Cropping & Coordinate Mapping
# ============================================================


def crop_to_roi(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.10,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Crop an image to the ROI bounding box with optional margin.

    Args:
        image: PIL Image
        bbox: (x, y, w, h) bounding box
        margin: Fractional margin to add on each side (0.10 = 10%)

    Returns:
        (cropped_image, adjusted_bbox) — the adjusted bbox reflects clamping
        to image boundaries
    """
    img_w, img_h = image.size
    x, y, w, h = bbox

    # Add margin
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)

    adjusted_bbox = (x1, y1, x2 - x1, y2 - y1)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped, adjusted_bbox


def map_anomaly_to_original(
    anomaly_map_roi: np.ndarray,
    roi_bbox: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
    background_value: float = 0.0,
) -> np.ndarray:
    """Map a ROI-space anomaly map back to original image coordinates.

    The anomaly_map_roi covers only the cropped ROI region. This function
    resizes it to fit within the ROI bbox and places it on a full-size
    canvas matching the original image dimensions.

    Args:
        anomaly_map_roi: Anomaly map in ROI space (H_roi, W_roi), already normalized
        roi_bbox: (x, y, w, h) of the ROI in original image coordinates
        original_size: (H, W) of the original image
        background_value: Value to fill outside the ROI region

    Returns:
        Full-size anomaly map (H, W) with ROI values placed correctly
    """
    h_orig, w_orig = original_size
    x, y, w, h = roi_bbox

    # Create full-size canvas
    canvas = np.full((h_orig, w_orig), background_value, dtype=np.float32)

    # Resize anomaly map to ROI dimensions
    anomaly_resized = cv2.resize(
        anomaly_map_roi.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Clamp ROI coordinates to be safe
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_orig - x)
    h = min(h, h_orig - y)

    # Place in canvas
    canvas[y:y + h, x:x + w] = anomaly_resized[:h, :w]

    return canvas


def expand_bbox_square(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Expand a bbox to be square (max of w, h), clamped to image boundaries.

    This is useful before feeding a cropped ROI into the feature extractor
    which expects square inputs.
    """
    img_w, img_h = image_size
    x, y, w, h = bbox
    max_dim = max(w, h)
    cx, cy = x + w // 2, y + h // 2

    half = max_dim // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, x1 + max_dim)
    y2 = min(img_h, y1 + max_dim)

    return (x1, y1, x2 - x1, y2 - y1)


# ============================================================
# Orchestrator
# ============================================================


class ObjectLocalizer:
    """Orchestrates object localization with automatic strategy selection.

    Usage:
        localizer = ObjectLocalizer()
        result = localizer.localize(image, saliency_map=saliency)
        cropped, adj_bbox = crop_to_roi(image, result.bbox, margin=0.10)
    """

    def __init__(
        self,
        default_method: str = "auto",
        min_area_ratio: float = 0.05,
        saliency_threshold_method: str = "otsu",
        contour_edge_low: int = 50,
        contour_edge_high: int = 150,
    ):
        self.default_method = default_method
        self.min_area_ratio = min_area_ratio
        self.saliency_threshold_method = saliency_threshold_method
        self.contour_edge_low = contour_edge_low
        self.contour_edge_high = contour_edge_high

    def localize(
        self,
        image: Image.Image,
        saliency_map: Optional[np.ndarray] = None,
        method: Optional[str] = None,
    ) -> LocalizationResult:
        """Run object localization on an image.

        Args:
            image: PIL Image
            saliency_map: DINOv2 saliency map (optional, required for saliency/auto methods)
            method: Strategy override — "saliency", "contour", "auto", "manual", or "none"

        Returns:
            LocalizationResult
        """
        method = method or self.default_method
        img_np = np.array(image)
        img_shape = (image.height, image.width)

        if method == "none":
            return LocalizationResult(
                bbox=(0, 0, image.width, image.height),
                confidence=1.0,
                method="none",
            )

        if method == "saliency":
            if saliency_map is None:
                logger.warning("Saliency method requested but no saliency map provided, falling back to contour")
                return localize_contour(img_np, min_area_ratio=self.min_area_ratio)
            return localize_saliency(
                saliency_map, img_shape,
                threshold_method=self.saliency_threshold_method,
                min_area_ratio=self.min_area_ratio,
            )

        if method == "contour":
            return localize_contour(
                img_np,
                edge_low=self.contour_edge_low,
                edge_high=self.contour_edge_high,
                min_area_ratio=self.min_area_ratio,
            )

        if method == "auto":
            # Try saliency first, fall back to contour
            if saliency_map is not None:
                result = localize_saliency(
                    saliency_map, img_shape,
                    threshold_method=self.saliency_threshold_method,
                    min_area_ratio=self.min_area_ratio,
                )
                if result.confidence >= 0.3 and result.method != "saliency_fallback":
                    return result
                logger.info("Saliency confidence low (%.2f), trying contour fallback", result.confidence)

            contour_result = localize_contour(
                img_np,
                edge_low=self.contour_edge_low,
                edge_high=self.contour_edge_high,
                min_area_ratio=self.min_area_ratio,
            )
            # Prefer contour if it found a tighter bbox
            if contour_result.method != "contour_fallback" and contour_result.area < result.area * 0.8:
                return contour_result
            return result if saliency_map is not None else contour_result

        raise ValueError(f"Unknown localization method: {method}")

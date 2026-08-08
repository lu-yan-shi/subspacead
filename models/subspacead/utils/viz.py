"""Visualization utilities for anomaly detection results.

Provides rendering functions for heatmap overlay, side-by-side comparison,
bbox annotation, ROI localization visualization, and image saving helpers.
"""

from __future__ import annotations

import logging
import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class VizMode(str, Enum):
    """可视化模式枚举"""
    OVERLAY = "overlay"          # 原图叠加热力图
    SIDE_BY_SIDE = "side_by_side"  # 左边原图，右边叠加
    BBOX = "bbox"                # 原图缺陷区域画红色框


# ============================================================
# Core Drawing Primitives
# ============================================================

def ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    """Ensures a numpy image array is 3-channel RGB."""
    if len(img_np.shape) == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.shape[2] == 1:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    return img_np


def create_heatmap(anom_map_norm_float: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Converts a 0–1 float anomaly map to an 8-bit BGR colormap image."""
    anom_map_u8 = (np.clip(anom_map_norm_float, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(anom_map_u8, colormap)


def add_text_to_image(img_np: np.ndarray, text: str,
                      pos: Tuple[int, int] = (10, 25),
                      font_scale: float = 0.7,
                      color: Tuple[int, int, int] = (255, 255, 255),
                      thickness: int = 2) -> np.ndarray:
    """Adds standardized text overlay to an image (returns copy)."""
    return cv2.putText(
        img_np.copy(), text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
    )


def find_defect_bbox(anom_map_norm: np.ndarray, threshold: float = 0.5,
                     min_area: float = 100) -> Optional[Tuple[int, int, int, int]]:
    """
    从归一化的异常热力图中找到缺陷区域的边界框。

    Args:
        anom_map_norm: 归一化到 0-1 的异常热力图
        threshold: 二值化阈值
        min_area: 最小缺陷面积（像素）

    Returns:
        (x, y, w, h) 边界框坐标，如果没有检测到缺陷则返回 None
    """
    anom_map_u8 = (anom_map_norm * 255).astype(np.uint8)
    _, binary = cv2.threshold(anom_map_u8, int(threshold * 255), 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)


# ============================================================
# Rendering Functions (return numpy arrays, no file I/O)
# ============================================================

def render_visualization(
    img: Image.Image,
    anom_map: np.ndarray,
    viz_mode: str = "overlay",
    score: Optional[float] = None,
    bbox_threshold: float = 0.5,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    渲染可视化结果，返回 BGR numpy 数组。

    Args:
        img: PIL 原图
        anom_map: 归一化到 0-1 的异常热力图 (H, W)
        viz_mode: "overlay" / "side_by_side" / "bbox"
        score: 异常分数（可选）
        bbox_threshold: 缺陷检测阈值
        roi_bbox: 目标定位框 (x, y, w, h)，在原图上绘制绿色框

    Returns:
        BGR numpy 数组
    """
    h, w = anom_map.shape
    img_np = np.array(img.resize((w, h)))
    img_np_rgb = ensure_rgb(img_np)
    heatmap = create_heatmap(anom_map)

    if viz_mode == "overlay":
        result = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        if roi_bbox is not None:
            rx, ry, rw, rh = roi_bbox
            cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        if score is not None:
            result = add_text_to_image(result, f"Score: {score:.4f}")

    elif viz_mode == "side_by_side":
        left = add_text_to_image(img_np_rgb, "Original")
        right = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        if roi_bbox is not None:
            rx, ry, rw, rh = roi_bbox
            cv2.rectangle(right, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        right = add_text_to_image(right, "With Heatmap")
        result = np.hstack([left, right])

    elif viz_mode == "bbox":
        result = img_np_rgb.copy()
        if roi_bbox is not None:
            rx, ry, rw, rh = roi_bbox
            cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(result, "ROI", (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        bbox = find_defect_bbox(anom_map, threshold=bbox_threshold)
        if bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(result, f"Defect: {bw}x{bh}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if score is not None:
            result = add_text_to_image(result, f"Score: {score:.4f}")

    else:
        raise ValueError(f"Unknown viz_mode: {viz_mode}")

    return result


def draw_roi_bbox(
    img_np: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    """在原图上绘制 ROI 定位框（绿色）。"""
    result = img_np.copy()
    x, y, w, h = bbox
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(result, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return result


# ============================================================
# File-Saving Functions
# ============================================================

def save_overlay_for_intro(
    path: str,
    img: Image.Image,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    kernel_size: int = 5,
    overlay_intensity: float = 0.4,
):
    """
    Saves a denoised, blended overlay for introductory figures.
    Assumes anom_map is a 0-1 normalized float array.
    """
    img_h, img_w = anom_map.shape
    img_np = np.array(img.resize((img_w, img_h)))
    img_np = ensure_rgb(img_np)

    anom_map_u8 = (anom_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)
    try:
        _, binary_mask = cv2.threshold(
            anom_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except cv2.error:
        binary_mask = np.zeros_like(anom_map_u8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    denoised_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    denoised_mask = cv2.dilate(denoised_mask, kernel, iterations=1)
    overlay = cv2.addWeighted(
        img_np, (1.0 - overlay_intensity), heatmap, overlay_intensity, 0
    )
    mask_3d = ensure_rgb(denoised_mask)
    final_image = np.where(mask_3d > 0, overlay, img_np)
    vis_dir = Path(outdir) / "intro_overlays" / category
    vis_dir.mkdir(parents=True, exist_ok=True)

    p = Path(path)
    unique_filename = f"{p.parent.name}_{p.name}"
    out_path = vis_dir / unique_filename
    Image.fromarray(final_image).save(out_path)


def save_visualization(
    path: str,
    img: Image.Image,
    gt_mask: np.ndarray,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    vis_idx: int,
    saliency_mask: Optional[np.ndarray] = None,
):
    """Saves a 2×2 multi-panel visualization (Original, GT, Map, Saliency/Overlay)."""
    target_shape = (anom_map.shape[1], anom_map.shape[0])
    target_shape_hw = (anom_map.shape[0], anom_map.shape[1])

    img_np = np.array(img.resize(target_shape))
    img_np_rgb = ensure_rgb(img_np)
    heatmap = create_heatmap(anom_map)

    if gt_mask.shape != target_shape_hw:
        logger.warning(
            f"GT shape {gt_mask.shape} != Anom map shape {target_shape_hw}. Resizing GT."
        )
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            target_shape,
            interpolation=cv2.INTER_NEAREST,
        )
    gt_mask_vis = ensure_rgb((gt_mask * 255).astype(np.uint8))
    panel1 = add_text_to_image(img_np_rgb, "Original")
    panel2 = add_text_to_image(gt_mask_vis, "Ground Truth")
    panel3 = add_text_to_image(heatmap, "Anomaly Map")

    if saliency_mask is not None:
        saliency_mask_u8 = (saliency_mask * 255).astype(np.uint8)
        saliency_mask_vis = ensure_rgb(saliency_mask_u8)
        panel4 = add_text_to_image(saliency_mask_vis, "Saliency Mask (FG)")
    else:
        overlay = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        panel4 = add_text_to_image(overlay, "Overlay")
    combined_img = np.vstack([np.hstack([panel1, panel2]), np.hstack([panel3, panel4])])

    vis_dir = Path(outdir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{category}_example_{vis_idx}.png"
    Image.fromarray(combined_img).save(out_path)


def save_custom_visualization(
    path: str,
    img: Image.Image,
    anom_map: np.ndarray,
    outdir: str,
    viz_mode: str = "overlay",
    score: Optional[float] = None,
    bbox_threshold: float = 0.5,
):
    """保存自定义可视化结果到文件。"""
    final_image = render_visualization(
        img, anom_map,
        viz_mode=viz_mode, score=score, bbox_threshold=bbox_threshold,
    )

    vis_dir = Path(outdir) / viz_mode
    vis_dir.mkdir(parents=True, exist_ok=True)

    p = Path(path) if isinstance(path, str) else Path(f"image_{hash(path)}")
    unique_filename = f"{p.stem}_{viz_mode}.png"
    out_path = vis_dir / unique_filename
    Image.fromarray(final_image).save(out_path)
    return str(out_path)


# ============================================================
# Backward-compatible aliases (deprecated)
# ============================================================

def _warn_deprecated(old_name: str, new_name: str):
    warnings.warn(
        f"'{old_name}' is deprecated, use '{new_name}' instead.",
        DeprecationWarning, stacklevel=2,
    )


def _find_defect_bbox(anom_map_norm, threshold=0.5):
    _warn_deprecated("_find_defect_bbox", "find_defect_bbox")
    return find_defect_bbox(anom_map_norm, threshold)


def _create_heatmap(anom_map_norm_float):
    _warn_deprecated("_create_heatmap", "create_heatmap")
    return create_heatmap(anom_map_norm_float)


def _ensure_rgb(img_np):
    _warn_deprecated("_ensure_rgb", "ensure_rgb")
    return ensure_rgb(img_np)


def _add_text_to_image(img_np, text):
    _warn_deprecated("_add_text_to_image", "add_text_to_image")
    return add_text_to_image(img_np, text)

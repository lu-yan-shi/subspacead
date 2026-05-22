from PIL import Image
import numpy as np
import cv2
import os
import logging
from typing import Optional, Tuple
from pathlib import Path
from enum import Enum


class VizMode(str, Enum):
    """可视化模式枚举"""
    OVERLAY = "overlay"  # 原图叠加热力图
    SIDE_BY_SIDE = "side_by_side"  # 左边原图，右边叠加
    BBOX = "bbox"  # 原图缺陷区域画红色框


def _add_text_to_image(img_np: np.ndarray, text: str) -> np.ndarray:
    """Adds standardized white text to the top-left corner of an image."""
    return cv2.putText(
        img_np.copy(),
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _find_defect_bbox(anom_map_norm: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    """
    从归一化的异常热力图中找到缺陷区域的边界框
    
    Args:
        anom_map_norm: 归一化到 0-1 的异常热力图
        threshold: 二值化阈值
        
    Returns:
        (x, y, w, h) 边界框坐标，如果没有检测到缺陷则返回 None
    """
    # 二值化
    anom_map_u8 = (anom_map_norm * 255).astype(np.uint8)
    _, binary = cv2.threshold(anom_map_u8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # 形态学操作去噪
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 如果最大轮廓面积太小，认为没有缺陷
    if cv2.contourArea(largest_contour) < 100:  # 最小面积阈值
        return None
    
    # 获取边界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, w, h)


def _ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    """Ensures a numpy image array is 3-channel RGB."""
    if len(img_np.shape) == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    return img_np


def _create_heatmap(anom_map_norm_float: np.ndarray) -> np.ndarray:
    """Converts a 0-1 float anomaly map to an 8-bit JET colormap."""
    anom_map_u8 = (anom_map_norm_float * 255).astype(np.uint8)
    return cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)


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
    img_np = _ensure_rgb(img_np)

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
    mask_3d = _ensure_rgb(denoised_mask)
    final_image = np.where(mask_3d > 0, overlay, img_np)
    vis_dir = Path(outdir) / "intro_overlays" / category
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique filename like "contamination_001.png"
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
    """Saves a 2x2 multi-panel visualization (Original, GT, Map, Saliency/Overlay)."""
    target_shape = (anom_map.shape[1], anom_map.shape[0])  # (W, H)
    target_shape_hw = (anom_map.shape[0], anom_map.shape[1])  # (H, W)

    img_np = np.array(img.resize(target_shape))
    img_np_rgb = _ensure_rgb(img_np)

    heatmap = _create_heatmap(anom_map)

    if gt_mask.shape != target_shape_hw:
        logging.warning(
            f"GT shape {gt_mask.shape} != Anom map shape {target_shape_hw}. Resizing GT."
        )
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8),
            target_shape,
            interpolation=cv2.INTER_NEAREST,
        )
    gt_mask_vis = _ensure_rgb((gt_mask * 255).astype(np.uint8))
    panel1 = _add_text_to_image(img_np_rgb, "Original")
    panel2 = _add_text_to_image(gt_mask_vis, "Ground Truth")
    panel3 = _add_text_to_image(heatmap, "Anomaly Map")

    if saliency_mask is not None:
        saliency_mask_u8 = (saliency_mask * 255).astype(np.uint8)
        saliency_mask_vis = _ensure_rgb(saliency_mask_u8)
        panel4 = _add_text_to_image(saliency_mask_vis, "Saliency Mask (FG)")
    else:
        overlay = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        panel4 = _add_text_to_image(overlay, "Overlay")
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
    """
    保存自定义可视化结果，支持多种模式
    
    Args:
        path: 图像路径（用于生成文件名）
        img: PIL Image 对象
        anom_map: 归一化到 0-1 的异常热力图
        outdir: 输出目录
        viz_mode: 可视化模式 ("overlay", "side_by_side", "bbox")
        score: 异常分数（可选，用于添加到图像上）
        bbox_threshold: 缺陷检测阈值（仅 bbox 模式使用）
    """
    img_h, img_w = anom_map.shape
    img_np = np.array(img.resize((img_w, img_h)))
    img_np_rgb = _ensure_rgb(img_np)
    
    # 创建热力图
    heatmap = _create_heatmap(anom_map)
    
    # 根据模式生成不同的可视化
    if viz_mode == "overlay":
        # 模式 1: 原图叠加热力图
        final_image = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        if score is not None:
            final_image = _add_text_to_image(final_image, f"Score: {score:.4f}")
        
    elif viz_mode == "side_by_side":
        # 模式 2: 左边原图，右边叠加
        left_panel = _add_text_to_image(img_np_rgb, "Original")
        overlay = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        right_panel = _add_text_to_image(overlay, "With Heatmap")
        final_image = np.hstack([left_panel, right_panel])
        
    elif viz_mode == "bbox":
        # 模式 3: 原图 + 缺陷区域红色框
        final_image = img_np_rgb.copy()
        bbox = _find_defect_bbox(anom_map, threshold=bbox_threshold)
        
        if bbox is not None:
            x, y, w, h = bbox
            # 画红色矩形框
            cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # 添加标签
            label = f"Defect Area: {w}x{h}"
            cv2.putText(
                final_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        
        if score is not None:
            final_image = _add_text_to_image(final_image, f"Score: {score:.4f}")
    else:
        raise ValueError(f"Unknown visualization mode: {viz_mode}")
    
    # 保存结果
    vis_dir = Path(outdir) / viz_mode
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    p = Path(path) if isinstance(path, str) else Path(f"image_{hash(path)}")
    unique_filename = f"{p.stem}_{viz_mode}.png"
    out_path = vis_dir / unique_filename
    Image.fromarray(final_image).save(out_path)
    
    return str(out_path)

"""
Training-Free 图结构 double-check — LayoutAD 方案的无监督实现。

核心思想 (与 LayoutAD 一致):
  正常产品 → 稳定的空间布局 → 参考图结构
  测试图像 → 提取区域构图 → 与参考图对比 → 结构异常分数

方法:
  1. 从模板图像构建参考图 (节点=区域, 边=空间关系)
  2. 从测试图像构建查询图
  3. 多维度图结构对比: 节点特征分布 / 边模式 / 拓扑统计量
  4. 综合异常分数

零额外依赖, 无需训练, 即插即用。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def _extract_regions(
    anomaly_map: np.ndarray,
    min_area: int = 200,
    max_regions: int = 32,
) -> List[Dict]:
    """从异常热力图提取候选区域 (与 inference.py 一致)。"""
    H, W = anomaly_map.shape
    regions = []
    seen = np.zeros((H, W), dtype=bool)
    for thr in [0.25, 0.45, 0.65]:
        if len(regions) >= max_regions:
            break
        binary = (anomaly_map > thr).astype(np.uint8)
        binary[seen] = 0
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        for i in range(1, n_labels):
            if len(regions) >= max_regions:
                break
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            mask = (labels == i).astype(np.uint8)
            seen[mask > 0] = True
            regions.append({"bbox": (x, y, w, h), "area": area, "mask": mask,
                           "anomaly_mean": float(anomaly_map[mask > 0].mean())})
    return regions


def _region_centroid(r: Dict) -> Tuple[float, float]:
    x, y, w, h = r["bbox"]
    return x + w / 2, y + h / 2


def _build_graph_fingerprint(
    regions: List[Dict],
    img_w: int, img_h: int,
) -> Optional[Dict]:
    """从区域列表构建图结构指纹。

    指纹包含:
      - num_nodes: 节点数
      - node_features: 每个节点的 [cx, cy, w, h, area] 归一化
      - edge_distances: 所有节点对之间的距离分布
      - adjacency_pattern: k-NN 邻接模式特征
      - layout_stats: 布局统计量 (质心偏移, 包围盒, 密度)
    """
    n = len(regions)
    if n < 2:
        return None

    # ── Node features ──
    node_feats = np.zeros((n, 5), dtype=np.float32)
    for i, r in enumerate(regions):
        cx, cy = _region_centroid(r)
        node_feats[i] = [cx / img_w, cy / img_h,
                         r["bbox"][2] / img_w, r["bbox"][3] / img_h,
                         r["area"] / (img_w * img_h)]

    # ── Pairwise distances ──
    dists = []
    angles = []
    for i in range(n):
        for j in range(i + 1, n):
            cxi, cyi = _region_centroid(regions[i])
            cxj, cyj = _region_centroid(regions[j])
            d = np.sqrt((cxi - cxj)**2 + (cyi - cyj)**2) / max(img_w, img_h)
            dists.append(d)
            angles.append(np.arctan2(cyj - cyi, cxj - cxi))

    dists = np.array(dists) if dists else np.zeros(1)
    angles = np.array(angles) if angles else np.zeros(1)

    # ── k-NN adjacency ──
    k = min(5, n - 1)
    adj_degrees = []
    for i in range(n):
        cxi, cyi = _region_centroid(regions[i])
        neigh_dists = []
        for j in range(n):
            if i == j:
                continue
            cxj, cyj = _region_centroid(regions[j])
            neigh_dists.append(np.sqrt((cxi-cxj)**2 + (cyi-cyj)**2))
        neigh_dists.sort()
        top_k = neigh_dists[:k]
        adj_degrees.append(len(top_k))
        # Store mean distance to k nearest neighbors
        if top_k:
            dists = np.append(dists, np.mean(top_k) / max(img_w, img_h))

    # ── Layout statistics ──
    centroids = np.array([_region_centroid(r) for r in regions])
    centroid_mean = centroids.mean(axis=0)
    centroid_std = centroids.std(axis=0)
    bbox_x = [r["bbox"][0] for r in regions]
    bbox_y = [r["bbox"][1] for r in regions]
    bbox_x2 = [r["bbox"][0] + r["bbox"][2] for r in regions]
    bbox_y2 = [r["bbox"][1] + r["bbox"][3] for r in regions]
    total_area = sum(r["area"] for r in regions) / (img_w * img_h)
    spatial_spread = (max(bbox_x2) - min(bbox_x)) * (max(bbox_y2) - min(bbox_y)) / (img_w * img_h)

    return {
        "num_nodes": n,
        "node_feats": node_feats,
        "dist_mean": float(dists.mean()),
        "dist_std": float(dists.std()),
        "dist_median": float(np.median(dists)),
        "angle_entropy": float(-np.sum(np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0] / len(angles) *
                                       np.log(np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0] / len(angles) + 1e-8))),
        "adj_mean_deg": float(np.mean(adj_degrees)) if adj_degrees else 0,
        "centroid_mean": centroid_mean / np.array([img_w, img_h]),
        "centroid_std": centroid_std / np.array([img_w, img_h]),
        "total_area": total_area,
        "spatial_spread": spatial_spread,
        "density": n / max(spatial_spread, 0.01),
    }


def _compare_fingerprints(ref: Dict, qry: Dict) -> Dict:
    """对比参考图和查询图的指纹，输出差异分数。

    多维度对比:
      1. 节点数差异
      2. 距离分布差异 (Wasserstein-like)
      3. 布局统计差异 (质心漂移, 散布变化)
      4. 拓扑差异 (角度熵, 密度)
    """
    scores = {}

    # 1. 节点数变化率
    n_ref, n_qry = ref["num_nodes"], qry["num_nodes"]
    scores["node_count_change"] = abs(n_qry - n_ref) / max(n_ref, 1)

    # 2. 距离分布差异
    scores["dist_mean_diff"] = abs(qry["dist_mean"] - ref["dist_mean"]) / max(ref["dist_mean"], 0.001)
    scores["dist_std_diff"] = abs(qry["dist_std"] - ref["dist_std"]) / max(ref["dist_std"], 0.001)

    # 3. 布局统计
    centroid_drift = np.linalg.norm(qry["centroid_mean"] - ref["centroid_mean"])
    scores["centroid_drift"] = float(centroid_drift)
    scores["spread_change"] = abs(qry["spatial_spread"] - ref["spatial_spread"]) / max(ref["spatial_spread"], 0.01)
    scores["area_change"] = abs(qry["total_area"] - ref["total_area"]) / max(ref["total_area"], 0.001)

    # 4. 拓扑
    scores["angle_entropy_diff"] = abs(qry["angle_entropy"] - ref["angle_entropy"])
    scores["density_change"] = abs(qry["density"] - ref["density"]) / max(ref["density"], 0.01)

    # 加权融合 → 最终结构异常分
    weights = {
        "node_count_change": 0.15,
        "dist_mean_diff": 0.20,
        "dist_std_diff": 0.15,
        "centroid_drift": 0.10,
        "spread_change": 0.20,
        "area_change": 0.10,
        "angle_entropy_diff": 0.05,
        "density_change": 0.05,
    }
    structural_score = sum(scores[k] * weights[k] for k in weights)
    # Sigmoid normalize to [0, 1]
    structural_score = 1.0 / (1.0 + np.exp(-5 * (structural_score - 0.3)))
    scores["structural_score"] = float(structural_score)

    return scores


class GraphStructureChecker:
    """Training-Free 图结构 double-check。

    使用方式:
        >>> checker = GraphStructureChecker()
        >>> checker.build_reference(template_images, detector)
        >>> result = checker.check(test_image, anomaly_map)
        >>> print(result["structural_score"], result["is_anomaly"])
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reference_fingerprints: List[Dict] = []
        self.reference_aggregated: Optional[Dict] = None
        self.is_built = False

    def build_reference(
        self,
        template_images: List[Image.Image],
        anomaly_maps: Optional[List[np.ndarray]] = None,
    ):
        """从模板图像构建参考图指纹。

        Args:
            template_images: 正常模板图像
            anomaly_maps: 对应的异常热力图 (SubspaceAD 输出), 为 None 时自动生成空白图
        """
        self.reference_fingerprints = []
        img_w, img_h = template_images[0].size

        for i, img in enumerate(template_images):
            if anomaly_maps and i < len(anomaly_maps):
                amap = anomaly_maps[i]
            else:
                # 无热力图时用空白图 (退化为均匀网格检测)
                amap = np.ones((img_h, img_w), dtype=np.float32) * 0.1

            regions = _extract_regions(amap)
            if len(regions) >= 2:
                fp = _build_graph_fingerprint(regions, img_w, img_h)
                if fp:
                    self.reference_fingerprints.append(fp)

        if self.reference_fingerprints:
            # 聚合多张模板图 → 平均参考指纹
            keys = ["num_nodes", "dist_mean", "dist_std", "dist_median",
                    "angle_entropy", "adj_mean_deg", "total_area",
                    "spatial_spread", "density"]
            agg = {}
            for k in keys:
                vals = [fp[k] for fp in self.reference_fingerprints if k in fp]
                agg[k] = float(np.mean(vals)) if vals else 0.0
            # centroid 取平均
            cmeans = np.array([fp["centroid_mean"] for fp in self.reference_fingerprints])
            cstds = np.array([fp["centroid_std"] for fp in self.reference_fingerprints])
            agg["centroid_mean"] = cmeans.mean(axis=0) if len(cmeans) else np.zeros(2)
            agg["centroid_std"] = cstds.mean(axis=0) if len(cstds) else np.zeros(2)
            self.reference_aggregated = agg
            self.is_built = True
            logger.info(
                "Graph reference built: %d template(s) → %d fingerprints",
                len(template_images), len(self.reference_fingerprints),
            )
        else:
            logger.warning("Failed to build graph reference — no valid regions extracted.")

    def check(
        self,
        anomaly_map: np.ndarray,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """对单张测试图像执行图结构 double-check。

        Args:
            anomaly_map: SubspaceAD 异常热力图
            img_size: 原图 (W, H)

        Returns:
            {structural_score, is_anomaly, detail, ...}
        """
        result = {
            "structural_score": 0.0,
            "is_anomaly": False,
            "available": False,
            "detail": {},
        }

        if not self.is_built or self.reference_aggregated is None:
            return result

        H, W = anomaly_map.shape
        if img_size:
            W, H = img_size

        regions = _extract_regions(anomaly_map)
        if len(regions) < 2:
            # Not enough regions — likely no anomaly signal
            return {**result, "available": True, "node_count": len(regions)}

        qry_fp = _build_graph_fingerprint(regions, W, H)
        if qry_fp is None:
            return {**result, "available": True, "node_count": len(regions)}

        # Compare to reference
        detail = _compare_fingerprints(self.reference_aggregated, qry_fp)
        score = detail["structural_score"]

        return {
            "structural_score": round(score, 6),
            "is_anomaly": score > self.threshold,
            "available": True,
            "node_count": len(regions),
            "detail": detail,
        }

    def fuse_scores(
        self,
        subspace_score: float,
        graph_result: Dict,
        subspace_weight: float = 0.6,
    ) -> Dict:
        """融合 SubspaceAD 和 图结构 分数。"""
        if graph_result.get("available"):
            graph_score = graph_result["structural_score"]
            final = subspace_weight * subspace_score + (1 - subspace_weight) * graph_score
            return {
                "final_score": round(final, 6),
                "is_anomaly": final > 0.5,
                "subspace_score": subspace_score,
                "structural_score": graph_score,
                "fusion_method": "weighted",
            }
        return {
            "final_score": subspace_score,
            "is_anomaly": subspace_score > 0.5,
            "subspace_score": subspace_score,
            "structural_score": 0.0,
            "fusion_method": "subspace_only",
        }

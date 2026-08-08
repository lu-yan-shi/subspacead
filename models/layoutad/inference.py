"""
LayoutAD 推理封装 — GNN + Transformer 图结构异常检测 double-check。

═══════════════════════════════════════════════════════════════
官方 LayoutAD 完整流水线 (需额外依赖):
  1. Mask2Former (Detectron2) → 全景分割，提取实例/语义区域
  2. CLIP → 编码每个区域的视觉特征 + 文本特征
  3. GraphConstructor → 构建图 (几何特征 + 边关系)
  4. LayoutAD GNN → 节点/边异常打分 → 结构级异常分数

轻量 fallback (零额外依赖):
  1. SubspaceAD 异常热力图 → 多级阈值提取可疑区域
  2. DINOv2 patch 特征 → 替代 CLIP 视觉特征
  3. k-NN 空间构图 → 简化的图结构
  4. LayoutAD GNN → 异常分数

安装完整流水线 (可选):
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  pip install 'git+https://github.com/facebookresearch/Mask2Former.git'
  pip install 'git+https://github.com/openai/CLIP.git'

模型权重: 训练后放置于 weights/layoutad_checkpoint.pth
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch_geometric.data import Data, Batch

from .model import LayoutAD
from .aggregate import score_aggregate
from .tools import norm_per_graph, build_pixel_score_map

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "layoutad_checkpoint.pth")

# ── Optional: Mask2Former + CLIP for full pipeline ──
_HAS_MASK2FORMER = False
_HAS_CLIP = False
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    _HAS_MASK2FORMER = True
except ImportError:
    pass

try:
    import clip
    _HAS_CLIP = True
except ImportError:
    pass


# ================================================================
# LayoutADInference
# ================================================================

class LayoutADInference:
    """LayoutAD 推理器 — GNN 结构级异常检测 double-check。

    两种工作模式:
    - **full**: Mask2Former 分割 + CLIP 特征 (高质量, 需额外依赖)
    - **light**: SubspaceAD 热力图提取区域 + DINOv2 特征 (零额外依赖)

    使用方式:
        >>> engine = LayoutADInference()
        >>> result = engine.check(image=img, anomaly_map=amap, ...)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        d_model: int = 256,
        num_enc_layers: int = 4,
        num_gnn_layers: int = 2,
        confidence_threshold: float = 0.5,
        # Full pipeline
        mask2former_config: Optional[str] = None,
        mask2former_weights: Optional[str] = None,
        clip_model: str = "ViT-B/32",
    ):
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.d_model = d_model
        self.num_enc_layers = num_enc_layers
        self.num_gnn_layers = num_gnn_layers
        self.confidence_threshold = confidence_threshold

        self.model: Optional[LayoutAD] = None
        self._model_loaded = False
        self._model_available = os.path.exists(self.checkpoint_path)

        # Full pipeline components (lazy init)
        self._segmentor = None       # Mask2Former DefaultPredictor
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._pipeline_mode: str = "light"  # "full" or "light"

        # Try to init full pipeline
        if _HAS_MASK2FORMER and mask2former_config and mask2former_weights:
            self._init_segmentor(mask2former_config, mask2former_weights)
        if _HAS_CLIP:
            self._init_clip(clip_model)

        if self._segmentor is not None and self._clip_model is not None:
            self._pipeline_mode = "full"
            logger.info("LayoutAD pipeline: FULL (Mask2Former + CLIP)")
        else:
            logger.info("LayoutAD pipeline: LIGHT (SubspaceAD region extraction)")

        # Load model
        if self._model_available:
            self._load_model()
        else:
            logger.warning("LayoutAD 权重未找到 (%s)，double-check 将自动跳过。", self.checkpoint_path)

    @property
    def available(self) -> bool:
        return self._model_loaded

    @property
    def pipeline_mode(self) -> str:
        return self._pipeline_mode

    # ================================================================
    # Full pipeline: Mask2Former + CLIP
    # ================================================================

    def _init_segmentor(self, config_path: str, weights_path: str):
        """初始化 Mask2Former 全景分割器。"""
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.data import MetadataCatalog

            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.DEVICE = str(self.device)
            self._segmentor = DefaultPredictor(cfg)
            # COCO panoptic metadata
            try:
                self._stuff_ids = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
            except Exception:
                self._stuff_ids = set()
            logger.info("Mask2Former segmentor initialized.")
        except Exception as e:
            logger.warning("Mask2Former init failed: %s. Falling back to light mode.", e)
            self._segmentor = None

    def _init_clip(self, model_name: str = "ViT-B/32"):
        """初始化 CLIP 视觉编码器。"""
        try:
            import clip
            self._clip_model, self._clip_preprocess = clip.load(model_name, device=self.device)
            self._clip_model.eval()
            self._clip_tokenizer = clip
            logger.info("CLIP model loaded: %s", model_name)
        except Exception as e:
            logger.warning("CLIP init failed: %s", e)
            self._clip_model = None

    def _segment_full(self, image: np.ndarray) -> List[Dict]:
        """使用 Mask2Former 做全景分割，提取区域 (节点)。"""
        if self._segmentor is None:
            return []

        H, W = image.shape[:2]
        outputs = self._segmentor(image)
        panoptic_seg = outputs["panoptic_seg"]  # (H, W) tensor
        segments_info = outputs["instances"] if "instances" in outputs else []

        # panoptic_seg[0] = semantic, panoptic_seg[1] = instance id
        if isinstance(panoptic_seg, torch.Tensor):
            panoptic_seg = panoptic_seg.cpu().numpy()

        nodes = []
        for seg in segments_info:
            seg_id = seg.get("id", len(nodes))
            is_thing = seg.get("isthing", True)
            area = seg.get("area", 0)
            category_id = seg.get("category_id", 0)

            # Instance mask
            mask = (panoptic_seg == seg_id).astype(np.uint8)
            if mask.sum() < 500:
                continue

            # Bbox + geometric features
            x, y, w, h = cv2.boundingRect(mask)
            M = cv2.moments(mask)
            cx = (M["m10"] / M["m00"]) if M["m00"] > 0 else x + w / 2
            cy = (M["m01"] / M["m00"]) if M["m00"] > 0 else y + h / 2
            cx_n, cy_n = cx / W, cy / H
            w_n, h_n = w / W, h / H
            area_norm = area / (W * H)
            aspect = w / max(h, 1)
            size_ratio = area_norm / max(w_n * h_n, 1e-6)
            hu = cv2.HuMoments(M).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

            # CLIP visual feature (crop & encode)
            vis_feat = None
            if self._clip_model is not None:
                try:
                    crop = image[y:y+h, x:x+w]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else crop
                    crop_pil = Image.fromarray(crop_rgb)
                    crop_tensor = self._clip_preprocess(crop_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        vis_feat = self._clip_model.encode_image(crop_tensor).squeeze(0).cpu()
                except Exception:
                    vis_feat = torch.zeros(512)

            nodes.append({
                "id": seg_id,
                "category_id": category_id,
                "is_thing": is_thing,
                "x": x / W, "y": y / H, "w": w_n, "h": h_n,
                "cx": cx_n, "cy": cy_n,
                "area": area_norm,
                "aspect_ratio": aspect,
                "size_ratio": size_ratio,
                "hu_moments": hu.tolist(),
                "vis_feat": vis_feat if vis_feat is not None else torch.zeros(512),
                "mask": mask,
            })

        return nodes

    # ================================================================
    # Light pipeline: SubspaceAD-based region extraction
    # ================================================================

    def _extract_regions_light(
        self,
        attention_map: np.ndarray,
        anomaly_map: np.ndarray,
        min_area: int = 200,
        max_regions: int = 32,
    ) -> List[Dict]:
        """从 SubspaceAD 异常热力图中提取候选区域。"""
        H, W = anomaly_map.shape
        regions = []
        thresholds = [0.3, 0.5, 0.7]
        seen = np.zeros((H, W), dtype=bool)

        for thr in thresholds:
            if len(regions) >= max_regions:
                break
            binary = (anomaly_map > thr).astype(np.uint8)
            binary[seen] = 0
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
            for i in range(1, n_labels):
                if len(regions) >= max_regions:
                    break
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                mask = (labels == i).astype(np.uint8)
                seen[mask > 0] = True
                attn_val = float(attention_map[mask > 0].mean()) if attention_map is not None else 0.5
                regions.append({
                    "bbox": (x, y, w, h), "area": area, "mask": mask,
                    "attention": attn_val, "anomaly_mean": float(anomaly_map[mask > 0].mean()),
                })
        return regions

    # ================================================================
    # Graph construction (shared)
    # ================================================================

    def _build_geo_features(self, nodes: List[Dict], img_w: int, img_h: int) -> torch.Tensor:
        """构建几何特征 [N, 17] — 兼容 full 和 light 两种节点格式。"""
        feats = []
        for n in nodes:
            if "cx" in n:  # full pipeline format (already normalized)
                cx, cy = n["cx"], n["cy"]
                w_n, h_n = n["w"], n["h"]
                area_n = n["area"]
                aspect = n["aspect_ratio"]
                size_ratio = n["size_ratio"]
                hu = n.get("hu_moments", [0]*7)
            else:  # light pipeline format (bbox in pixels)
                x, y, w, h = n["bbox"]
                cx, cy = (x + w/2) / img_w, (y + h/2) / img_h
                w_n, h_n = w / img_w, h / img_h
                area_n = n["area"] / (img_w * img_h)
                aspect = w / max(h, 1)
                size_ratio = area_n / max(w_n * h_n, 1e-6)
                hu = [0]*7
            geo = torch.tensor([
                cx, cy, w_n, h_n, area_n, aspect, size_ratio,
                *hu[:7], 0, 0, 0,
            ], dtype=torch.float32)
            geo = (geo - geo.mean()) / (geo.std() + 1e-6)
            feats.append(geo)
        return torch.stack(feats) if feats else torch.zeros(0, 17)

    def _build_vis_features(
        self, nodes: List[Dict], dino_features: Optional[torch.Tensor],
        img_w: int, img_h: int, f_h: int, f_w: int,
    ) -> torch.Tensor:
        """构建视觉特征 [N, d_model]"""
        feats = []
        for n in nodes:
            if "vis_feat" in n and n["vis_feat"] is not None and n["vis_feat"].numel() > 0:
                # Full pipeline: CLIP feature
                vf = n["vis_feat"].float()
            elif dino_features is not None and dino_features.numel() > 0:
                # Light pipeline: DINOv2 patch pooling
                n_patches = dino_features.shape[0]
                fmap_size = int(np.sqrt(n_patches))
                f_h_actual = f_w_actual = fmap_size

                if "bbox" in n:
                    x, y, w, h = n["bbox"]
                else:
                    x = n["x"] * img_w; y = n["y"] * img_h
                    w = n["w"] * img_w; h = n["h"] * img_h
                fx0 = max(0, int(x / img_w * f_w_actual))
                fy0 = max(0, int(y / img_h * f_h_actual))
                fx1 = min(f_w_actual, int((x + w) / img_w * f_w_actual) + 1)
                fy1 = min(f_h_actual, int((y + h) / img_h * f_h_actual) + 1)

                indices = []
                for py in range(fy0, fy1):
                    for px in range(fx0, fx1):
                        indices.append(py * f_w_actual + px)
                if indices:
                    idx_t = torch.tensor(indices, dtype=torch.long)
                    vf = dino_features[idx_t].mean(dim=0)
                else:
                    vf = torch.zeros(self.d_model)
                vf = F.normalize(vf.float(), dim=0)
            else:
                vf = torch.zeros(self.d_model)
            feats.append(vf)
        return torch.stack(feats) if feats else torch.zeros(0, self.d_model)

    def _build_edges(
        self, nodes: List[Dict], x_vis: torch.Tensor, img_w: int, img_h: int, k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建边: k-NN 空间图 + 几何/视觉边特征。"""
        num = len(nodes)
        edges_i, edges_geo, edges_vis = [], [], []

        # Compute centers
        centers = []
        for n in nodes:
            if "cx" in n:
                centers.append((n["cx"] * img_w, n["cy"] * img_h))
            else:
                x, y, w, h = n["bbox"]
                centers.append((x + w/2, y + h/2))

        for i in range(num):
            cxi, cyi = centers[i]
            dists = []
            for j in range(num):
                if i == j:
                    continue
                cxj, cyj = centers[j]
                d = np.sqrt((cxi-cxj)**2 + (cyi-cyj)**2) / max(img_w, img_h)
                dists.append((j, d))
            dists.sort(key=lambda t: t[1])
            for j, d in dists[:min(k, len(dists))]:
                cxj, cyj = centers[j]
                dx = (cxj - cxi) / img_w
                dy = (cyj - cyi) / img_h
                angle = np.arctan2(dy, dx)

                # Area and IoU (bbox-based)
                def _bbox(n):
                    return n["bbox"] if "bbox" in n else (n["x"]*img_w, n["y"]*img_h, n["w"]*img_w, n["h"]*img_h)
                bxi, byi, bwi, bhi = _bbox(nodes[i])
                bxj, byj, bwj, bhj = _bbox(nodes[j])
                ai, aj = bwi*bhi, bwj*bhj
                area_ratio = min(ai, aj) / max(ai, aj, 1)
                w_ratio = min(bwi, bwj) / max(bwi, bwj, 1)
                h_ratio = min(bhi, bhj) / max(bhi, bhj, 1)
                inter_w = max(0, min(bxi+bwi, bxj+bwj) - max(bxi, bxj))
                inter_h = max(0, min(byi+bhi, byj+bhj) - max(byi, byj))
                iou = (inter_w * inter_h) / max(ai + aj - inter_w * inter_h, 1)

                edges_i.append((i, j))
                e_geo = torch.tensor([dx, dy, d, np.sin(angle), np.cos(angle), w_ratio, h_ratio, area_ratio, iou], dtype=torch.float32)
                edges_geo.append(e_geo)

                if x_vis.size(0) > max(i, j):
                    sim = F.cosine_similarity(x_vis[i].unsqueeze(0), x_vis[j].unsqueeze(0), dim=-1).item()
                else:
                    sim = 0.0
                edges_vis.append(torch.tensor([sim, 0.0], dtype=torch.float32))

        if edges_i:
            ei = torch.tensor(edges_i, dtype=torch.long).t().contiguous()
            eg = torch.stack(edges_geo)
            ev = torch.stack(edges_vis)
            eg = (eg - eg.mean(dim=0)) / (eg.std(dim=0) + 1e-6)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
            eg = torch.zeros((0, 9)); ev = torch.zeros((0, 2))
        return ei, eg, ev

    def build_graph(
        self,
        nodes: List[Dict],
        img_size: Tuple[int, int],
        dino_features: Optional[torch.Tensor] = None,
        fmap_hw: Optional[Tuple[int, int]] = None,
    ) -> Data:
        """从节点列表构建 PyG Data 图。

        Args:
            nodes: 节点列表 (full pipeline: Mask2Former 节点; light: 热力图区域)
            img_size: (W, H) 原图尺寸
            dino_features: DINOv2 patch 特征 (light mode)
            fmap_hw: (H_f, W_f) 特征图尺寸
        """
        img_w, img_h = img_size
        num_nodes = len(nodes)
        if num_nodes == 0:
            return Data(x_geo=torch.zeros(0, 17), x_vis=torch.zeros(0, self.d_model),
                       edge_index=torch.zeros((2, 0), dtype=torch.long),
                       edge_attr_geo=torch.zeros(0, 9), edge_attr_vis=torch.zeros(0, 2), num_nodes=0)

        f_h, f_w = fmap_hw or (1, 1)
        x_geo = self._build_geo_features(nodes, img_w, img_h)
        x_vis = self._build_vis_features(nodes, dino_features, img_w, img_h, f_h, f_w)
        ei, eg, ev = self._build_edges(nodes, x_vis, img_w, img_h)

        return Data(x_geo=x_geo, x_vis=x_vis, edge_index=ei, edge_attr_geo=eg,
                    edge_attr_vis=ev, num_nodes=num_nodes)

    # ================================================================
    # Model loading
    # ================================================================

    def _load_model(self):
        logger.info("Loading LayoutAD from %s ...", self.checkpoint_path)
        try:
            self.model = LayoutAD(d_model=self.d_model, num_enc_layers=self.num_enc_layers,
                                  num_gnn_layers=self.num_gnn_layers).to(self.device)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self._model_loaded = True
            logger.info("LayoutAD loaded (device=%s).", self.device)
        except Exception as e:
            logger.error("LayoutAD load failed: %s", e)
            self._model_loaded = False

    # ================================================================
    # Main check method
    # ================================================================

    @torch.no_grad()
    def check(
        self,
        image: Image.Image,
        dino_features: Optional[torch.Tensor] = None,
        attention_map: Optional[np.ndarray] = None,
        anomaly_map: Optional[np.ndarray] = None,
        feature_map_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """执行 LayoutAD double-check。

        Args:
            image: PIL 图像
            dino_features: DINOv2 patch 特征 (light mode)
            attention_map: 注意力图 (light mode)
            anomaly_map: 异常热力图 (light mode)
            feature_map_hw: DINO 特征图 (H_f, W_f)

        Returns:
            {available, layout_score, is_anomaly, node_count, node_scores, graph_built}
        """
        result = {"available": self.available, "layout_score": 0.0, "is_anomaly": False,
                  "node_count": 0, "node_scores": [], "graph_built": False, "pipeline_mode": self._pipeline_mode}
        if not self.available:
            return result

        img_w, img_h = image.size
        nodes = []

        # ── Extract nodes ──
        if self._pipeline_mode == "full" and self._segmentor is not None:
            img_np = np.array(image.convert("RGB"))
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            nodes = self._segment_full(img_np)
        else:
            if anomaly_map is None and attention_map is None:
                return result
            amap = anomaly_map if anomaly_map is not None else attention_map
            attn = attention_map if attention_map is not None else amap
            if amap.shape != (img_h, img_w):
                amap = cv2.resize(amap, (img_w, img_h))
            if attn.shape != (img_h, img_w):
                attn = cv2.resize(attn, (img_w, img_h))
            nodes = self._extract_regions_light(attn, amap)

        result["node_count"] = len(nodes)
        if len(nodes) < 2:
            return result

        # ── Build graph ──
        try:
            graph = self.build_graph(nodes, (img_w, img_h), dino_features, feature_map_hw)
        except Exception as e:
            logger.warning("Graph build failed: %s", e)
            return result

        if graph.num_nodes < 2:
            return result
        result["graph_built"] = True

        # ── Inference ──
        try:
            batch = Batch.from_data_list([graph]).to(self.device)
            output = self.model(batch)
            scores = output["score"]
            result["node_scores"] = scores.cpu().tolist()
            if scores.numel() > 0:
                layout_score = float(scores.max().item())
                result["layout_score"] = float(torch.sigmoid(torch.tensor(layout_score)).item())
            result["is_anomaly"] = result["layout_score"] > self.confidence_threshold
        except Exception as e:
            logger.error("LayoutAD inference failed: %s", e)

        return result

    # ================================================================
    # Score fusion
    # ================================================================

    def fuse_scores(self, subspace_score: float, layout_result: Dict, subspace_weight: float = 0.6) -> Dict:
        """融合 SubspaceAD 和 LayoutAD 分数。"""
        if layout_result["available"] and layout_result["graph_built"]:
            layout_score = layout_result["layout_score"]
            final = subspace_weight * subspace_score + (1 - subspace_weight) * layout_score
            return {
                "final_score": round(final, 6),
                "is_anomaly": final > 0.5,
                "subspace_score": subspace_score,
                "layout_score": round(layout_score, 6),
                "layout_available": True,
                "fusion_method": "weighted",
                "pipeline_mode": layout_result.get("pipeline_mode", "light"),
            }
        return {
            "final_score": subspace_score,
            "is_anomaly": subspace_score > 0.5,
            "subspace_score": subspace_score,
            "layout_score": 0.0,
            "layout_available": False,
            "fusion_method": "subspace_only",
            "pipeline_mode": "n/a",
        }

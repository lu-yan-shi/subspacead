"""
SubspaceAD 异常检测器 - 可迁移的工业质检模块
基于 DINOv2 特征和 PCA 子空间建模的少样本异常检测
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from PIL import Image
import numpy as np
import cv2
import torch

# 导入核心组件 - 使用相对导入支持直接使用
import sys
from pathlib import Path

# 获取当前文件所在目录
current_dir = Path(__file__).parent
src_dir = current_dir / "src"

# 添加 src 到路径
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from subspacead.core.extractor import FeatureExtractor
    from subspacead.core.pca import PCAModel
    from subspacead.post_process.scoring import calculate_anomaly_scores
    from subspacead.utils.common import min_max_norm
except ImportError as e:
    raise ImportError(
        f"无法导入 subspacead 包。请确保文件结构完整。错误详情：{e}"
    )

logger = logging.getLogger(__name__)


class SubspaceAnomalyDetector:
    """
    基于 SubspaceAD 的异常检测器
    
    特性:
        - 少样本学习：仅需 1-2 张正常图像即可训练
        - 自适应层选择：根据模型深度自动选择最优特征层
        - 多尺度特征聚合：融合多层 DINOv2 特征
        - 实时推理：单次前向传播即可完成检测
        
    示例:
        >>> detector = SubspaceAnomalyDetector()
        >>> detector.train_normal(template_images=["normal_1.jpg", "normal_2.jpg"])
        >>> results = detector.detect(test_images=["test_1.jpg", "test_2.jpg"])
        >>> for result in results:
        ...     print(f"{result['image_name']}: {result['anomaly_score']:.4f}")
    """
    
    def __init__(
        self,
        model_ckpt: str = None,  # 默认使用本地模型
        image_res: int = 512,
        pca_ev: float = 0.99,
        device: Optional[str] = None,
        use_clahe: bool = False,
        score_method: str = "reconstruction",
        drop_k: int = 0,
    ):
        """
        初始化异常检测器
        
        Args:
            model_ckpt: DINOv2 模型路径或 HuggingFace 模型名
                       如果为 None，默认使用本地 models/dinov2-small 目录
            image_res: 输入图像分辨率（正方形）
            pca_ev: PCA 保留方差比例 (0-1)
            device: 计算设备 ("cuda"/"cpu")，默认自动选择
            use_clahe: 是否使用 CLAHE 增强
            score_method: 异常分数计算方法 ("reconstruction"/"mahalanobis"/"euclidean"/"cosine")
            drop_k: 丢弃前 k 个主成分（用于去除正常变异）
        """
        # 如果没有指定模型路径，使用本地模型
        if model_ckpt is None:
            # 获取当前文件所在目录
            current_dir = Path(__file__).parent
            local_model_path = current_dir / "models" / "dinov2-small"
            
            if local_model_path.exists():
                self.model_ckpt = str(local_model_path)
                logger.info(f"使用本地模型：{self.model_ckpt}")
            else:
                # 如果本地模型不存在，使用远程模型
                self.model_ckpt = "facebook/dinov2-small"
                logger.info(f"本地模型未找到，使用远程模型：{self.model_ckpt}")
        else:
            self.model_ckpt = model_ckpt
        self.image_res = image_res
        self.pca_ev = pca_ev
        self.use_clahe = use_clahe
        self.score_method = score_method
        self.drop_k = drop_k
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # 初始化组件
        self.extractor: Optional[FeatureExtractor] = None
        self.pca_params: Optional[Dict] = None
        self.is_trained = False
        
        # 层配置（根据模型深度自动选择）
        self.layers_config = {
            "giant": list(range(-8, 0)),   # 40+ layers
            "large": list(range(-6, 0)),   # 24+ layers
            "base": list(range(-4, 0)),    # <24 layers
        }
        self.agg_method = "mean"  # 特征聚合方式
    
    def _get_model_layers(self, num_hidden_layers: int) -> List[int]:
        """根据模型深度获取使用的层索引"""
        if num_hidden_layers >= 40:
            layers = self.layers_config["giant"]
        elif num_hidden_layers >= 24:
            layers = self.layers_config["large"]
        else:
            layers = self.layers_config["base"]
        
        logger.info(f"Model has {num_hidden_layers} layers, using: {layers}")
        return layers
    
    def train(
        self,
        template_images: Union[List[str], List[Image.Image]],
        verbose: bool = True,
    ) -> Dict:
        """
        使用正常图像训练 PCA 模型
        
        Args:
            template_images: 正常图像路径列表或 PIL Image 对象列表
            verbose: 是否打印日志
            
        Returns:
            训练信息字典
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("🔧 Training SubspaceAD Model")
            logger.info("=" * 60)
        
        # 1. 加载特征提取器
        if verbose:
            logger.info("\n📦 Loading DINOv2 model...")
        self.extractor = FeatureExtractor(self.model_ckpt)
        
        # 2. 准备图像
        if isinstance(template_images[0], str):
            imgs = [Image.open(p).convert("RGB") for p in template_images]
        else:
            imgs = template_images
        
        if verbose:
            logger.info(f"Loaded {len(imgs)} template images")
        
        # 3. 获取模型配置
        extractor_model_cfg = self.extractor.model.config
        num_hidden_layers = extractor_model_cfg.num_hidden_layers
        layers = self._get_model_layers(num_hidden_layers)
        
        # 4. 提取特征
        if verbose:
            logger.info("\n🔍 Extracting features from templates...")
        
        tokens, (h_p, w_p), _ = self.extractor.extract_tokens(
            imgs,
            self.image_res,
            layers,
            self.agg_method,
            docrop=False,
            use_clahe=self.use_clahe,
        )
        
        b, _, _, c = tokens.shape
        feature_dim = c
        tokens_reshaped = tokens.reshape(b * h_p * w_p, c)
        
        if verbose:
            logger.info(f"  Feature dimension: {feature_dim}")
            logger.info(f"  Token grid: {h_p} x {w_p}")
            logger.info(f"  Total tokens: {b * h_p * w_p}")
        
        # 5. 拟合 PCA
        if verbose:
            logger.info("\n📊 Training PCA model...")
        
        def feature_generator():
            yield tokens_reshaped
        
        pca_model = PCAModel(k=None, ev=self.pca_ev, whiten=False)
        self.pca_params = pca_model.fit(
            feature_generator,
            feature_dim,
            total_tokens=b * h_p * w_p,
            num_batches=1,
        )
        
        self.is_trained = True
        
        if verbose:
            logger.info(f"  PCA components: {self.pca_params['k']}")
            logger.info("✅ Training complete!")
            logger.info("=" * 60)
        
        return {
            "pca_components": self.pca_params["k"],
            "feature_dim": feature_dim,
            "grid_size": (h_p, w_p),
            "num_templates": len(imgs),
        }
    
    def detect(
        self,
        test_images: Union[List[str], List[Image.Image]],
        save_dir: Optional[str] = None,
        save_visualizations: bool = False,
        viz_mode: str = "overlay",
        bbox_threshold: float = 0.5,
        top_k_ratio: float = 0.01,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        对测试图像进行异常检测
        
        Args:
            test_images: 测试图像路径列表或 PIL Image 对象列表
            save_dir: 结果保存目录（如果为 None 则不保存）
            save_visualizations: 是否保存可视化结果
            viz_mode: 可视化模式 ("overlay", "side_by_side", "bbox")
                     - "overlay": 原图叠加热力图
                     - "side_by_side": 左边原图，右边叠加
                     - "bbox": 原图缺陷区域画红色框
            bbox_threshold: 缺陷检测阈值（仅 bbox 模式使用）
            top_k_ratio: 计算图像级分数时使用的 top-k 比例
            verbose: 是否打印日志
            
        Returns:
            检测结果列表，每项包含：
                - image_name: 图像名称
                - anomaly_score: 异常分数（标量）
                - anomaly_map: 归一化的异常热力图（np.ndarray）
                - overlay_path: 叠加图路径（如果保存）
                - anomaly_map_path: 热力图路径（如果保存）
                - viz_path: 自定义可视化路径（如果保存）
        """
        if not self.is_trained:
            raise RuntimeError("请先调用 train() 方法训练模型")
        
        if verbose:
            logger.info("\n🔍 Detecting anomalies...")
        
        # 准备输出目录
        if save_dir and save_visualizations:
            os.makedirs(save_dir, exist_ok=True)
            # 不预先创建任何子目录，按需创建
        
        # 加载图像
        if isinstance(test_images[0], str):
            imgs = [Image.open(p).convert("RGB") for p in test_images]
            img_names = [Path(p).stem for p in test_images]
        else:
            imgs = test_images
            img_names = [f"image_{i}" for i in range(len(test_images))]
        
        results = []
        
        for i, test_img in enumerate(imgs):
            test_name = img_names[i]
            
            # 提取特征
            tokens_test, (h_p_test, w_p_test), _ = self.extractor.extract_tokens(
                [test_img],
                self.image_res,
                self._get_model_layers(self.extractor.model.config.num_hidden_layers),
                self.agg_method,
                docrop=False,
                use_clahe=self.use_clahe,
            )
            
            b_test, _, _, c_test = tokens_test.shape
            tokens_test_reshaped = tokens_test.reshape(
                b_test * h_p_test * w_p_test, c_test
            )
            
            # 计算异常分数
            scores = calculate_anomaly_scores(
                tokens_test_reshaped,
                self.pca_params,
                method=self.score_method,
                drop_k=self.drop_k,
            )
            
            # 重塑为热力图
            anomaly_map = scores.reshape(h_p_test, w_p_test)
            
            # 后处理：双线性插值到原图大小
            anomaly_map_cv = cv2.resize(
                anomaly_map.astype(np.float32),
                test_img.size,
                interpolation=cv2.INTER_LINEAR
            )
            
            # 归一化到 0-1
            anomaly_map_normalized = min_max_norm(anomaly_map_cv)
            
            # 计算图像级异常分数（top-k% 均值）
            flat_scores = anomaly_map_normalized.flatten()
            k = max(1, int(len(flat_scores) * top_k_ratio))
            top_k_mean = np.mean(np.sort(flat_scores)[-k:])
            
            result = {
                "image_name": test_name,
                "anomaly_score": float(top_k_mean),
                "anomaly_map": anomaly_map_normalized,
            }
            
            # 保存结果 - 只保存用户选择的可视化模式
            if save_dir and save_visualizations:
                # 保存自定义可视化结果（唯一需要的可视化）
                try:
                    from subspacead.utils.viz import save_custom_visualization
                    viz_path = save_custom_visualization(
                        path=test_images[i] if isinstance(test_images[i], str) else f"image_{i}",
                        img=test_img,
                        anom_map=anomaly_map_normalized,
                        outdir=save_dir,
                        viz_mode=viz_mode,
                        score=float(top_k_mean),
                        bbox_threshold=bbox_threshold,
                    )
                    result["viz_path"] = viz_path
                except Exception as e:
                    logger.warning(f"Failed to save custom visualization: {e}")
            
            results.append(result)
            
            if verbose:
                status = "⚠️  ANOMALY" if result["anomaly_score"] > 0.3 else "✅ Normal"
                logger.info(
                    f"[{i+1}/{len(imgs)}] {test_name}: "
                    f"Score={result['anomaly_score']:.4f} ({status})"
                )
        
        if verbose:
            logger.info("=" * 60)
            logger.info("📊 Detection Summary:")
            for result in results:
                score = result["anomaly_score"]
                status = "⚠️  ANOMALY" if score > 0.3 else "✅ Normal"
                logger.info(f"{result['image_name']:20s} | Score: {score:.4f} | {status}")
            logger.info("=" * 60)
        
        return results
    
    def detect_single(
        self,
        image: Union[str, Image.Image],
        return_heatmap: bool = False,
        verbose: bool = True,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        单张图像快速检测
        
        Args:
            image: 图像路径或 PIL Image
            return_heatmap: 是否返回异常热力图
            verbose: 是否打印日志
            
        Returns:
            异常分数，或 (异常分数，热力图) 如果 return_heatmap=True
        """
        if isinstance(image, str):
            imgs = [Image.open(image).convert("RGB")]
        else:
            imgs = [image]
        
        results = self.detect(imgs, verbose=verbose)
        result = results[0]
        
        if return_heatmap:
            return result["anomaly_score"], result["anomaly_map"]
        else:
            return result["anomaly_score"]
    
    def set_threshold(self, threshold: float = 0.3):
        """
        设置异常判定阈值
        
        Args:
            threshold: 异常分数阈值（默认 0.3）
        """
        self.threshold = threshold
        logger.info(f"Anomaly threshold set to: {threshold}")
    
    def is_anomaly(self, score: float) -> bool:
        """
        判断分数是否为异常
        
        Args:
            score: 异常分数
            
        Returns:
            是否为异常
        """
        threshold = getattr(self, "threshold", 0.3)
        return score > threshold
    
    def export_model(self, save_path: str):
        """
        导出模型参数（用于部署）
        
        Args:
            save_path: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("请先训练模型")
        
        model_data = {
            "model_ckpt": self.model_ckpt,
            "image_res": self.image_res,
            "pca_ev": self.pca_ev,
            "pca_params": self.pca_params,
            "layers_config": self.layers_config,
            "agg_method": self.agg_method,
            "score_method": self.score_method,
            "drop_k": self.drop_k,
        }
        
        torch.save(model_data, save_path)
        logger.info(f"Model exported to: {save_path}")
    
    def load_model(self, load_path: str, init_extractor: bool = True):
        """
        加载模型参数
        
        Args:
            load_path: 模型路径
            init_extractor: 是否初始化特征提取器
        """
        model_data = torch.load(load_path, map_location=self.device)
        
        self.model_ckpt = model_data["model_ckpt"]
        self.image_res = model_data["image_res"]
        self.pca_ev = model_data["pca_ev"]
        self.pca_params = model_data["pca_params"]
        self.layers_config = model_data["layers_config"]
        self.agg_method = model_data["agg_method"]
        self.score_method = model_data["score_method"]
        self.drop_k = model_data["drop_k"]
        
        if init_extractor:
            self.extractor = FeatureExtractor(self.model_ckpt)
        
        self.is_trained = True
        logger.info(f"Model loaded from: {load_path}")


# 便捷函数
def create_detector(
    model: str = "facebook/dinov2-small",
    resolution: int = 512,
    **kwargs
) -> SubspaceAnomalyDetector:
    """
    快速创建检测器实例
    
    Args:
        model: 模型名称
        resolution: 分辨率
        **kwargs: 其他参数传递给 SubspaceAnomalyDetector
        
    Returns:
        SubspaceAnomalyDetector 实例
    """
    return SubspaceAnomalyDetector(
        model_ckpt=model,
        image_res=resolution,
        **kwargs
    )

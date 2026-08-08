"""LayoutAD — GNN + Transformer 图结构异常检测模块。

作为 SubspaceAD 的 double-check 方案：
- SubspaceAD 做像素级异常检测（DINOv2 + 记忆库）
- LayoutAD 做结构级验证（图神经网络 + 布局关系）

注意: LayoutAD/LayoutADInference 需要 torch_geometric (可选依赖)。
      graph_check 模块零依赖，始终可用。
"""

# Lazy imports — torch_geometric may not be installed
_HAS_TORCH_GEOMETRIC = False
try:
    from .model import LayoutAD
    from .inference import LayoutADInference
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    LayoutAD = None  # type: ignore
    LayoutADInference = None  # type: ignore

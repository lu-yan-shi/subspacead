try:
    from .detector import SubspaceAnomalyDetector
except ImportError:
    SubspaceAnomalyDetector = None  # type: ignore

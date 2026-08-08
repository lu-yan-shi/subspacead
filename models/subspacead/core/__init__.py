"""SubspaceAD core modules."""

from .localization import (
    ObjectLocalizer,
    LocalizationResult,
    localize_saliency,
    localize_contour,
    localize_manual,
    crop_to_roi,
    map_anomaly_to_original,
    expand_bbox_square,
)

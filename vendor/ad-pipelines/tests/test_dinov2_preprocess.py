from types import SimpleNamespace

import pytest
import torch

from ad_pipelines.models.model_dinov2 import DinoV2Model
from ad_pipelines.models.model_dinov2_with_register import DinoV2WithRegisterModel
from ad_pipelines.models.model_dinov3_vit import DinoV3ViTModel


@pytest.mark.parametrize(
    "model_class",
    (DinoV2Model, DinoV2WithRegisterModel, DinoV3ViTModel),
)
def test_preprocess_preserves_bf16_tensors_for_fast_processor(model_class) -> None:
    received = {}

    def processor(images, **kwargs):
        received["dtype"] = images.dtype
        received["kwargs"] = kwargs
        return {"pixel_values": images}

    model = SimpleNamespace(
        resolution=(224, 224),
        processor=processor,
    )
    images = torch.zeros((1, 3, 16, 16), dtype=torch.bfloat16)

    output = model_class.preprocess(model, images, do_resize=False)

    assert received["dtype"] == torch.bfloat16
    assert received["kwargs"]["return_tensors"] == "pt"
    assert output.dtype == torch.bfloat16

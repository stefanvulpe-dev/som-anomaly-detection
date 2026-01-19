import random
from typing import Tuple

import timm
import torch
import torchvision

from src.config import Config

_model: torch.nn.Module = None
_transform: torchvision.transforms.Compose = None
_selected_indexes: list[int] = None


def resolve_selected_indexes(cfg: Config) -> list[int]:
    """
    Resolve and return the selected feature indexes for SOM input.

    Args:
        cfg (Config): Configuration parameters including som_input_dimensions.

    Returns:
        list[int]: List of selected feature indexes.
    """
    global _selected_indexes, _model
    if _selected_indexes is None and _model is not None:
        with torch.no_grad():
            f1, f2, f3 = _model(torch.randn(1, 3, 224, 224, device=cfg.device))
            total_dimensions = f1.size(1) + f2.size(1) + f3.size(1)

        _selected_indexes = random.sample(
            range(total_dimensions),
            min(cfg.som_input_dimensions, total_dimensions),
        )

    return _selected_indexes


def get_backbone(
    cfg: Config,
) -> Tuple[
    torch.nn.Module,
    torchvision.transforms.Compose,
]:
    """
    Get the backbone model and corresponding image transform.

    Args:
        cfg (Config): Configuration parameters including backbone type and device.

    Returns:
        Tuple[torch.nn.Module, torchvision.transforms.Compose]: The backbone model and image transform.
    """
    global _model, _transform
    if _model is None:
        _model = timm.create_model(
            cfg.backbone,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2),
        ).to(cfg.device)
        _model.eval()

        data_config = timm.data.resolve_model_data_config(_model)
        _transform = timm.data.create_transform(
            **data_config, is_training=False
        )

    return _model, _transform

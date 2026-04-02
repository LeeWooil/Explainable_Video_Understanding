from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import FrozenVideoMAEBackbone


class LocalConceptHead(nn.Module):
    def __init__(self, in_dim: int, out_channels: int = 1) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VideoMAELocalizer(nn.Module):
    def __init__(self, backbone: FrozenVideoMAEBackbone, out_channels: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = LocalConceptHead(in_dim=self.backbone.model.embed_dim, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape {tuple(x.shape)}")
        if x.shape[1] == self.backbone.model.embed_dim:
            feature_map = x
        else:
            feature_map = self.backbone.forward_feature_map(x)
        logits = self.head(feature_map)
        return logits, feature_map

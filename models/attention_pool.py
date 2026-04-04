from __future__ import annotations

import torch
import torch.nn as nn


class ConceptGuidedAttentionPool(nn.Module):
    """Attention-weighted spatial pooling guided by concept activation maps.

    Uses the localizer's concept logits as soft attention masks on the
    backbone's feature map, producing a per-concept feature vector.

    Input:
        feature_map:    [B, D, T, H, W]  (backbone features)
        concept_logits: [B, C, T, H, W]  (localizer output)
    Output:
        pooled:         [B, C, D]         (per-concept attended features)
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, feature_map: torch.Tensor, concept_logits: torch.Tensor
    ) -> torch.Tensor:
        attn = torch.sigmoid(concept_logits)              # [B, C, T, H, W]
        attn_expanded = attn.unsqueeze(2)                  # [B, C, 1, T, H, W]
        feat_expanded = feature_map.unsqueeze(1)           # [B, 1, D, T, H, W]
        weighted = attn_expanded * feat_expanded           # [B, C, D, T, H, W]
        pooled = weighted.sum(dim=(3, 4, 5)) / (
            attn_expanded.sum(dim=(3, 4, 5)) + self.eps
        )                                                  # [B, C, D]
        return pooled

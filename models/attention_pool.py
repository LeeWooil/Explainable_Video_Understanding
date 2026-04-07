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

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, feature_map: torch.Tensor, concept_logits: torch.Tensor
    ) -> torch.Tensor:
        B, C, T, H, W = concept_logits.shape
        attn = torch.softmax(
            concept_logits.view(B, C, -1) / self.temperature, dim=-1
        ).view(B, C, T, H, W)                             # [B, C, T, H, W]
        attn_expanded = attn.unsqueeze(2)                  # [B, C, 1, T, H, W]
        feat_expanded = feature_map.unsqueeze(1)           # [B, 1, D, T, H, W]
        weighted = attn_expanded * feat_expanded           # [B, C, D, T, H, W]
        pooled = weighted.sum(dim=(3, 4, 5))               # [B, C, D]
        return pooled


class ConceptAwareSpatialPool(nn.Module):
    """Concept-aware spatial pooling producing a single global feature.

    Merges per-concept attention maps into one spatial importance map,
    then pools backbone features using that map.

    Input:
        feature_map:    [B, D, T, H, W]  (backbone features)
        concept_logits: [B, C, T, H, W]  (localizer output)
    Output:
        pooled:         [B, D]            (concept-aware global feature)
    """

    def __init__(self, temperature: float = 1.0, mode: str = "softmax") -> None:
        super().__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(
        self, feature_map: torch.Tensor, concept_logits: torch.Tensor
    ) -> torch.Tensor:
        B, C, T, H, W = concept_logits.shape
        D = feature_map.shape[1]
        S = T * H * W

        logit_flat = concept_logits.view(B, C, S)

        if self.mode == "sigmoid":
            attn = torch.sigmoid(logit_flat)                # [B, C, S]
        else:
            attn = torch.softmax(logit_flat / self.temperature, dim=-1)  # [B, C, S]

        # Merge: max across concepts -> single importance map
        importance, _ = attn.max(dim=1)                     # [B, S]

        # Re-normalize to sum to 1
        importance = importance / importance.sum(dim=1, keepdim=True)  # [B, S]

        # Weighted spatial pooling
        feat_flat = feature_map.view(B, D, S)               # [B, D, S]
        pooled = torch.einsum("bs, bds -> bd", importance, feat_flat)  # [B, D]
        return pooled

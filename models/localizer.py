from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FrozenVideoMAEBackbone


# ---------------------------------------------------------------------------
# Head type 1: 1x1x1 Conv3d (기존 baseline)
# ---------------------------------------------------------------------------
class LocalConceptHead(nn.Module):
    def __init__(self, in_dim: int, out_channels: int = 1) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Head type 2: 3x1x1 Temporal Conv
#   - 시간축 이웃 3개 timestep의 context를 반영
#   - 초기 block에서 뽑은 feature처럼 temporal 정보가 부족한 경우 보완 효과
#   - padding=(1,0,0)으로 temporal 차원 유지
# ---------------------------------------------------------------------------
class TemporalConvHead(nn.Module):
    def __init__(self, in_dim: int, out_channels: int = 1) -> None:
        super().__init__()
        self.temporal_conv = nn.Conv3d(
            in_dim, in_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False,
        )
        self.act = nn.GELU()
        self.proj = nn.Conv3d(in_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, H, W]
        x = self.act(self.temporal_conv(x))
        return self.proj(x)


# ---------------------------------------------------------------------------
# Head type 3: 3x3x3 Spatio-Temporal Conv
#   - 시간축 + 공간축 이웃 context를 함께 반영
#   - padding=(1,1,1)로 T, H, W 차원 유지
# ---------------------------------------------------------------------------
class SpatioTemporalConvHead(nn.Module):
    def __init__(self, in_dim: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim, in_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False,
        )
        self.act = nn.GELU()
        self.proj = nn.Conv3d(in_dim, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, H, W]
        x = self.act(self.conv(x))
        return self.proj(x)


# ---------------------------------------------------------------------------
# Head type 4: Cross-Attention
#   - 학습 가능한 concept query [C, D]가 patch token [T*H*W, D]에 attend
#   - attention weight [B, C, T*H*W] → reshape → [B, C, T, H, W] = localization map
#   - attn_logits를 직접 출력 (BCE loss와 호환되도록 sigmoid 전 raw logit)
# ---------------------------------------------------------------------------
class CrossAttentionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_channels: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_channels = out_channels  # = num concepts C
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        assert in_dim % num_heads == 0, f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})"

        # Learnable concept query embeddings [C, D]
        self.concept_queries = nn.Parameter(torch.randn(out_channels, in_dim) * 0.02)

        # Q, K, V projections
        self.q_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, in_dim, bias=False)

        # Output projection: attention pooled feature → scalar logit per concept per patch
        # 각 패치별 logit을 만들기 위해 concept query와 key의 dot product를 직접 사용
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.concept_queries, std=0.02)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, H, W]
        B, D, T, H, W = x.shape
        N = T * H * W  # number of patch tokens

        # Reshape to token sequence: [B, N, D]
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, N, D)

        # Project queries and keys
        # Q: [C, D] → [C, D] (shared across batch)
        # K: [B, N, D] → [B, N, D]
        Q = self.q_proj(self.concept_queries)  # [C, D]
        K = self.k_proj(tokens)                # [B, N, D]

        # Multi-head attention scores
        # Q: [C, num_heads, head_dim] → [num_heads, C, head_dim]
        # K: [B, N, num_heads, head_dim] → [B, num_heads, N, head_dim]
        Q = Q.view(self.out_channels, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, C, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)            # [B, num_heads, N, head_dim]

        # Expand Q for batch: [B, num_heads, C, head_dim]
        Q = Q.unsqueeze(0).expand(B, -1, -1, -1)

        # Attention logits: [B, num_heads, C, N]
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, num_heads, C, N]

        # Average across heads → [B, C, N]
        attn_logits = attn_logits.mean(dim=1)
        attn_logits = self.dropout(attn_logits)

        # Reshape to spatial: [B, C, T, H, W]
        logits = attn_logits.view(B, self.out_channels, T, H, W)
        return logits


# ---------------------------------------------------------------------------
# Factory function: head_type string → head module
# ---------------------------------------------------------------------------
def build_concept_head(
    head_type: str,
    in_dim: int,
    out_channels: int,
    num_heads: int = 4,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a concept localization head by type name.

    Args:
        head_type: One of "conv1x1x1", "conv3x1x1", "cross_attn"
        in_dim: Input feature dimension (e.g., 768 for ViT-Base)
        out_channels: Number of concepts C
        num_heads: Number of attention heads (only for cross_attn)
        dropout: Dropout rate (only for cross_attn)
    """
    if head_type == "conv1x1x1":
        return LocalConceptHead(in_dim=in_dim, out_channels=out_channels)
    elif head_type == "conv3x1x1":
        return TemporalConvHead(in_dim=in_dim, out_channels=out_channels)
    elif head_type == "conv3x3x3":
        return SpatioTemporalConvHead(in_dim=in_dim, out_channels=out_channels)
    elif head_type == "cross_attn":
        return CrossAttentionHead(
            in_dim=in_dim, out_channels=out_channels,
            num_heads=num_heads, dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown head_type '{head_type}'. "
            f"Choose from: conv1x1x1, conv3x1x1, conv3x3x3, cross_attn"
        )


# ---------------------------------------------------------------------------
# VideoMAELocalizer (기존 구조 유지, head_type 옵션 추가)
# ---------------------------------------------------------------------------
class VideoMAELocalizer(nn.Module):
    def __init__(
        self,
        backbone: FrozenVideoMAEBackbone,
        out_channels: int = 1,
        head_type: str = "conv1x1x1",
        num_heads: int = 4,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.head = build_concept_head(
            head_type=head_type,
            in_dim=self.backbone.model.embed_dim,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape {tuple(x.shape)}")
        if x.shape[1] == self.backbone.model.embed_dim:
            feature_map = x
        else:
            feature_map = self.backbone.forward_feature_map(x)
        logits = self.head(feature_map)
        return logits, feature_map

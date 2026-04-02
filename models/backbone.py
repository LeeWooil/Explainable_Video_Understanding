from __future__ import annotations

import torch
import torch.nn as nn

from . import modeling_finetune
from .model_loading import get_target_model, load_vmae_weight


def _build_vmae_fallback(args, device: torch.device) -> nn.Module:
    factory = getattr(modeling_finetune, args.backbone, None)
    if factory is None and args.backbone.startswith("vmae_"):
        factory = getattr(modeling_finetune, args.backbone.removeprefix("vmae_"), None)
    if factory is None:
        raise RuntimeError(f"Unknown model ({args.backbone})")
    model = factory(
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )
    patch_size = model.patch_embed.patch_size
    args.window_size = (
        args.num_frames // 2,
        args.input_size // patch_size[0],
        args.input_size // patch_size[1],
    )
    args.patch_size = patch_size
    if getattr(args, "finetune", ""):
        model = load_vmae_weight(model, args)
    model.to(device)
    return model


class FrozenVideoMAEBackbone(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        block_index: int,
        num_frames: int,
        tubelet_size: int,
        input_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.block_index = int(block_index)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.input_size = int(input_size)

        if not hasattr(self.model, "patch_embed") or not hasattr(self.model, "blocks"):
            raise TypeError("Expected a VideoMAE VisionTransformer-like model.")

        self.patch_size = int(self.model.patch_embed.patch_size[0])
        self.grid_t = self.num_frames // self.tubelet_size
        self.grid_h = self.input_size // self.patch_size
        self.grid_w = self.input_size // self.patch_size

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @classmethod
    def from_args(cls, args, device: torch.device) -> "FrozenVideoMAEBackbone":
        if str(args.backbone).startswith("vmae_"):
            model = _build_vmae_fallback(args, device)
            return cls(
                model=model,
                block_index=args.block_index,
                num_frames=args.num_frames * args.num_segments,
                tubelet_size=args.tubelet_size,
                input_size=args.input_size,
            )
        try:
            model, _ = get_target_model(args.backbone, device, args)
        except RuntimeError as exc:
            if "Unknown model" not in str(exc):
                raise
            model = _build_vmae_fallback(args, device)
        return cls(
            model=model,
            block_index=args.block_index,
            num_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            input_size=args.input_size,
        )

    @torch.no_grad()
    def forward_intermediate_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        bsz = x.shape[0]
        if getattr(self.model, "pos_embed", None) is not None:
            pos_embed = self.model.pos_embed.expand(bsz, -1, -1).type_as(x).to(x.device).clone().detach()
            x = x + pos_embed
        x = self.model.pos_drop(x)
        for idx, blk in enumerate(self.model.blocks):
            x = blk(x)
            if idx == self.block_index:
                return x
        raise IndexError(f"block_index={self.block_index} out of range for {len(self.model.blocks)} blocks.")

    @torch.no_grad()
    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_intermediate_tokens(x)
        bsz, num_tokens, dim = tokens.shape
        expected = self.grid_t * self.grid_h * self.grid_w
        if num_tokens != expected:
            raise ValueError(f"Unexpected token count: got {num_tokens}, expected {expected}.")
        feature_map = tokens.view(bsz, self.grid_t, self.grid_h, self.grid_w, dim)
        return feature_map.permute(0, 4, 1, 2, 3).contiguous()

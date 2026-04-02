from __future__ import annotations

from types import SimpleNamespace


def build_videomae_args(cli_args) -> SimpleNamespace:
    """Build the arg namespace expected by PCBEAR's VideoMAE loader."""
    return SimpleNamespace(
        backbone=cli_args.backbone,
        nb_classes=cli_args.nb_classes,
        num_frames=cli_args.num_frames,
        num_segments=cli_args.num_segments,
        tubelet_size=cli_args.tubelet_size,
        block_index=cli_args.block_index,
        fc_drop_rate=cli_args.fc_drop_rate,
        drop=cli_args.drop,
        drop_path=cli_args.drop_path,
        attn_drop_rate=cli_args.attn_drop_rate,
        use_checkpoint=cli_args.use_checkpoint,
        use_mean_pooling=cli_args.use_mean_pooling,
        init_scale=cli_args.init_scale,
        finetune=cli_args.finetune,
        input_size=cli_args.input_size,
        model_key=getattr(cli_args, "model_key", "model|module"),
        model_prefix=getattr(cli_args, "model_prefix", ""),
    )

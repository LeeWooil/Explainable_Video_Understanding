from __future__ import annotations

from collections import OrderedDict

import torch
from timm.models import create_model
from torchvision import models

from . import modeling_finetune  # noqa: F401


def _load_state_dict(model, state_dict, prefix: str = "", ignore_missing: str = "relative_position_index") -> None:
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, module_prefix=""):
        local_metadata = {} if metadata is None else metadata.get(module_prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            module_prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, module_prefix + name + ".")

    load(model, prefix=prefix)

    warned_missing = []
    ignored_missing = []
    for key in missing_keys:
        if any(ignore_key in key for ignore_key in ignore_missing.split("|")):
            ignored_missing.append(key)
        else:
            warned_missing.append(key)

    if warned_missing:
        print(f"Weights of {model.__class__.__name__} not initialized from pretrained model: {warned_missing}")
    if unexpected_keys:
        print(f"Weights from pretrained model not used in {model.__class__.__name__}: {unexpected_keys}")
    if ignored_missing:
        print(f"Ignored weights of {model.__class__.__name__} not initialized from pretrained model: {ignored_missing}")
    if error_msgs:
        print("\n".join(error_msgs))


def load_vmae_weight(model, args=None):
    if args is None or not getattr(args, "finetune", ""):
        return model

    if args.finetune.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location="cpu")

    print(f"Load ckpt from {args.finetune}")
    checkpoint_model = None
    for model_key in args.model_key.split("|"):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"Load state_dict by model_key = {model_key}")
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]

    new_dict = OrderedDict()
    for key in list(checkpoint_model.keys()):
        if key.startswith("backbone."):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith("encoder."):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        tubelet_frames = args.num_frames // model.patch_embed.tubelet_size
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // tubelet_frames) ** 0.5)
        new_size = int((num_patches // tubelet_frames) ** 0.5)
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, tubelet_frames, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                -1,
                tubelet_frames,
                new_size,
                new_size,
                embedding_size,
            )
            checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens.flatten(1, 3)), dim=1)

    _load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    print("\n\n\n*********** VMAE Load ***************")
    return model


def get_target_model(target_name, device, args=None):
    if target_name.startswith("vmae_"):
        model = create_model(
            target_name,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
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
        return model, None

    target_name_cap = target_name.replace("resnet", "ResNet")
    if target_name.endswith("_v2"):
        base_name = target_name[:-3]
        weights = eval(f"models.{target_name_cap[:-3]}_Weights.IMAGENET1K_V2")
        model = eval(f"models.{base_name}(weights=weights)").to(device)
        model.eval()
        return model, weights.transforms()

    if hasattr(models, target_name):
        weights = eval(f"models.{target_name_cap}_Weights.IMAGENET1K_V1")
        model = eval(f"models.{target_name}(weights=weights)").to(device)
        model.eval()
        return model, weights.transforms()

    raise RuntimeError(f"Unknown model ({target_name})")

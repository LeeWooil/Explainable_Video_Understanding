"""Extract and cache localizer outputs (concept_logits, feature_map) for CBM training.

Runs the frozen VideoMAELocalizer once on train/val splits and saves the raw outputs
so that downstream CBM training scripts can load them directly without re-running
the expensive backbone forward pass.

Output files per split:
    {mode}_concept_logits.pt  -- [N, C, T, H, W]
    {mode}_feature_map.pt     -- [N, D, T, H, W]
    {mode}_labels.pt          -- [N]
    {mode}_sample_ids.json    -- list of N strings
    extraction_meta.json      -- shapes, checkpoint path, block_index, timestamp
"""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.defaults import build_videomae_args
from datasets.local_video_dataset import LocalVideoDataset
from models.backbone import FrozenVideoMAEBackbone
from models.localizer import VideoMAELocalizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Extract and cache localizer features.")

# data
parser.add_argument("--anno-path", type=Path, required=True)
parser.add_argument("--val-anno-path", type=Path, required=True)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--val-data-root", type=Path, default=None)

# model
parser.add_argument("--backbone", type=str, default="vmae_vit_base_patch16_224")
parser.add_argument("--finetune", type=str, required=True)
parser.add_argument("--data-set", type=str, required=True)
parser.add_argument("--nb-classes", dest="nb_classes", type=int, required=True)
parser.add_argument("--num-concepts", type=int, required=True)
parser.add_argument("--block-index", type=int, default=6)
parser.add_argument("--num-frames", type=int, default=16)
parser.add_argument("--num-segments", type=int, default=1)
parser.add_argument("--sampling-rate", type=int, default=4)
parser.add_argument("--tubelet-size", type=int, default=2)
parser.add_argument("--input-size", type=int, default=224)
parser.add_argument("--patch-size", type=int, default=16)
parser.add_argument("--fc-drop-rate", dest="fc_drop_rate", type=float, default=0.0)
parser.add_argument("--drop", type=float, default=0.0)
parser.add_argument("--drop-path", dest="drop_path", type=float, default=0.1)
parser.add_argument("--attn-drop-rate", dest="attn_drop_rate", type=float, default=0.0)
parser.add_argument("--use-checkpoint", dest="use_checkpoint", action="store_true")
parser.add_argument("--use-mean-pooling", dest="use_mean_pooling", action="store_true")
parser.add_argument("--init-scale", dest="init_scale", type=float, default=0.001)
parser.add_argument("--model-key", dest="model_key", type=str, default="model|module")
parser.add_argument("--model-prefix", dest="model_prefix", type=str, default="")
parser.add_argument("--deterministic-spatial", dest="deterministic_spatial", action="store_true")
parser.add_argument("--localizer-ckpt", type=Path, required=True)
parser.add_argument("--head-type", type=str, default=None,
                    help="Localizer head type. Auto-detected from checkpoint if saved; "
                         "pass explicitly for old checkpoints without head_type key.")

# extraction
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)

parser.set_defaults(use_checkpoint=False, use_mean_pooling=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _collate_meta(batch):
    videos = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metas = [item[2] for item in batch]
    return videos, labels, metas


def make_loader(anno_path: Path, data_root: Path, args) -> DataLoader:
    dataset = LocalVideoDataset(
        anno_path=anno_path,
        data_root=data_root,
        data_set=args.data_set,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        deterministic=args.deterministic_spatial,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_meta,
    )
    return loader


def save_localizer_features(
    model: VideoMAELocalizer,
    dataloader: DataLoader,
    output_dir: Path,
    mode: str,
    device: torch.device,
) -> dict:
    """Run backbone and concept head, save concept_logits and full feature_map.

    concept_logits are produced from intermediate features (at block_index)
    via the concept head. feature_map is the output after all backbone blocks.

    Returns dict with tensor shapes for metadata.
    """
    logits_save = output_dir / f"{mode}_concept_logits.pt"
    fmap_save = output_dir / f"{mode}_feature_map.pt"
    labels_save = output_dir / f"{mode}_labels.pt"
    sids_save = output_dir / f"{mode}_sample_ids.json"

    # Skip if all outputs already exist
    if all(p.exists() for p in [logits_save, fmap_save, labels_save, sids_save]):
        print(f"[{mode}] All outputs already exist, skipping.")
        # Load shapes for metadata
        clogits = torch.load(logits_save, map_location="cpu", weights_only=True)
        fmap = torch.load(fmap_save, map_location="cpu", weights_only=True)
        shapes = {
            "concept_logits": list(clogits.shape),
            "feature_map": list(fmap.shape),
        }
        del clogits, fmap
        return shapes

    all_concept_logits = []
    all_feature_maps = []
    all_labels = []
    all_sample_ids = []

    with torch.no_grad():
        for videos, labels, metas in tqdm(dataloader, desc=f"extract-{mode}"):
            # intermediate_map: [B, D, T, H, W] at block_index (for concept head)
            # final_map:        [B, D, T, H, W] after all blocks (for projection)
            intermediate_map, final_map = model.backbone.forward_dual_feature_maps(
                videos.to(device)
            )
            concept_logits = model.head(intermediate_map)  # [B, C, T, H, W]

            all_concept_logits.append(concept_logits.cpu())
            all_feature_maps.append(final_map.cpu())
            all_labels.append(labels)
            all_sample_ids.extend(str(meta["sample_id"]) for meta in metas)

    concept_logits_cat = torch.cat(all_concept_logits)
    feature_map_cat = torch.cat(all_feature_maps)
    labels_cat = torch.cat(all_labels)

    print(f"[{mode}] concept_logits: {tuple(concept_logits_cat.shape)}")
    print(f"[{mode}] feature_map:    {tuple(feature_map_cat.shape)}")
    print(f"[{mode}] labels:         {tuple(labels_cat.shape)}")
    print(f"[{mode}] sample_ids:     {len(all_sample_ids)}")

    torch.save(concept_logits_cat, logits_save)
    torch.save(feature_map_cat, fmap_save)
    torch.save(labels_cat, labels_save)
    with open(sids_save, "w") as f:
        json.dump(all_sample_ids, f)

    shapes = {
        "concept_logits": list(concept_logits_cat.shape),
        "feature_map": list(feature_map_cat.shape),
    }

    # Free memory
    del all_concept_logits, all_feature_maps, all_labels
    del concept_logits_cat, feature_map_cat, labels_cat
    torch.cuda.empty_cache()

    return shapes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("=" * 50)
    print(" Localizer Feature Extraction")
    print(f" Dataset: {args.data_set}")
    print(f" Backbone: {args.backbone}")
    print(f" Block index: {args.block_index}")
    print(f" Num concepts: {args.num_concepts}")
    print(f" Localizer ckpt: {args.localizer_ckpt}")
    print(f" Device: {args.device}")
    print("=" * 50)

    setup_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build frozen localizer
    model_args = build_videomae_args(args)
    backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
    checkpoint = torch.load(args.localizer_ckpt, map_location="cpu", weights_only=True)
    ckpt_head_type = checkpoint.get("head_type", None) or args.head_type or "conv1x1x1"
    model = VideoMAELocalizer(backbone, out_channels=args.num_concepts, head_type=ckpt_head_type).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Extract for each split
    all_shapes = {}
    split_configs = {
        "train": (args.anno_path, args.data_root),
        "val": (args.val_anno_path, args.val_data_root or args.data_root),
    }

    for mode, (anno_path, data_root) in split_configs.items():
        loader = make_loader(anno_path, data_root, args)
        shapes = save_localizer_features(model, loader, output_dir, mode, device)
        all_shapes[mode] = shapes

    # Save extraction metadata
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "localizer_ckpt": str(args.localizer_ckpt),
        "backbone": args.backbone,
        "block_index": args.block_index,
        "num_concepts": args.num_concepts,
        "num_frames": args.num_frames,
        "sampling_rate": args.sampling_rate,
        "tubelet_size": args.tubelet_size,
        "input_size": args.input_size,
        "shapes": all_shapes,
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Saved to: {output_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

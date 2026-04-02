from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from configs.defaults import build_videomae_args
from datasets.local_video_dataset import LocalConceptVideoDataset
from models.backbone import FrozenVideoMAEBackbone
from models.localizer import VideoMAELocalizer
from utils.visualization import save_multiconcept_localization_previews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize multi-concept local predictions from a trained localizer.")
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pseudo-mask-root", type=Path, required=True)
    parser.add_argument("--localizer-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="vmae_vit_base_patch16_224")
    parser.add_argument("--finetune", type=str, required=True)
    parser.add_argument("--data-set", type=str, required=True)
    parser.add_argument("--nb-classes", dest="nb_classes", type=int, required=True)
    parser.add_argument("--num-concepts", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--view-mode", choices=["random", "center_uniform"], default="center_uniform")
    parser.add_argument("--target-cache-root", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--topk-concepts", type=int, default=3)
    parser.add_argument("--concept-source", choices=["target", "pred"], default="target")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-ids-json", type=Path, default=None)
    parser.add_argument("--save-gt-frames", action="store_true")
    parser.set_defaults(use_checkpoint=False, use_mean_pooling=True)
    return parser.parse_args()
def _collate_batch(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    targets = torch.stack([item[2] for item in batch], dim=0)
    metas = [item[3] for item in batch]
    return inputs, labels, targets, metas


def _load_sample_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of sample ids.")
    return {str(item) for item in data}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dataset = LocalConceptVideoDataset(
        anno_path=args.anno_path,
        data_root=args.data_root,
        data_set=args.data_set,
        pseudo_mask_root=args.pseudo_mask_root,
        tubelet_size=args.tubelet_size,
        patch_size=args.patch_size,
        target_cache_root=(args.target_cache_root / args.anno_path.stem) if args.target_cache_root is not None else None,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        deterministic=args.deterministic_spatial,
        view_mode=args.view_mode,
    )

    selected_ids = _load_sample_ids(args.sample_ids_json)
    if selected_ids is not None:
        selected_indices = []
        for idx, sample in enumerate(dataset.dataset_samples):
            sample_id = Path(sample).with_suffix("").as_posix()
            if sample_id in selected_ids:
                selected_indices.append(idx)
        dataset = Subset(dataset, selected_indices[: args.max_samples])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_batch,
    )

    model_args = build_videomae_args(args)
    backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
    model = VideoMAELocalizer(backbone, out_channels=args.num_concepts).to(device)
    checkpoint = torch.load(args.localizer_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    seen = 0
    progress = tqdm(loader, desc="visualize", dynamic_ncols=True)
    with torch.no_grad():
        for videos, _labels, targets, metas in progress:
            videos = videos.to(device, non_blocking=True)
            logits, _ = model(videos)
            batch_size = min(videos.shape[0], args.max_samples - seen)
            save_multiconcept_localization_previews(
                videos=videos[:batch_size].detach().cpu(),
                logits=logits[:batch_size].detach().cpu(),
                targets=targets[:batch_size].detach().cpu(),
                metas=metas[:batch_size],
                output_dir=args.output_dir,
                tag="multiconcept",
                max_samples=batch_size,
                threshold=args.threshold,
                topk_concepts=args.topk_concepts,
                concept_source=args.concept_source,
                concept_names=[str(i) for i in range(args.num_concepts)],
                save_gt_frames=args.save_gt_frames,
            )
            seen += batch_size
            if seen >= args.max_samples:
                break


if __name__ == "__main__":
    main()

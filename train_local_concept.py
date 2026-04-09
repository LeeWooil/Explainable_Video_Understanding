from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from configs.defaults import build_videomae_args
from datasets.local_video_dataset import LocalConceptVideoDataset, _build_sample_id
from models.backbone import FrozenVideoMAEBackbone
from models.localizer import VideoMAELocalizer
from utils.metrics import RunningBinaryLocalizationMetrics
from utils.visualization import save_localization_previews


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return not _is_distributed() or dist.get_rank() == 0


def _setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device(args.device)
        return device, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.device != "cuda":
        raise ValueError("Distributed training currently expects --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training but is not available.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return device, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a local concept head on frozen VideoMAE features.")
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pseudo-mask-root", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="vmae_vit_base_patch16_224")
    parser.add_argument("--finetune", type=str, required=True)
    parser.add_argument("--data-set", type=str, required=True)
    parser.add_argument("--nb-classes", dest="nb_classes", type=int, required=True)
    parser.add_argument("--num-concepts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--block-index", type=int, default=6)
    parser.add_argument("--head-type", type=str, default="conv1x1x1",
                        choices=["conv1x1x1", "conv3x1x1", "conv3x3x3", "cross_attn"],
                        help="Concept head type: conv1x1x1 (baseline), conv3x1x1 (temporal), conv3x3x3 (spatio-temporal), cross_attn")
    parser.add_argument("--head-num-heads", type=int, default=4,
                        help="Number of attention heads (only for cross_attn)")
    parser.add_argument("--head-dropout", type=float, default=0.0,
                        help="Dropout rate in concept head (only for cross_attn)")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-segments", type=int, default=1)
    parser.add_argument("--sampling-rate", type=int, default=4)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--eval-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/local_concept"))
    parser.add_argument("--save-preview-every", type=int, default=1)
    parser.add_argument("--preview-max-samples", type=int, default=4)
    parser.add_argument("--val-anno-path", type=Path, default=None)
    parser.add_argument("--val-data-root", type=Path, default=None)
    parser.add_argument("--early-stopping-patience", dest="early_stopping_patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", dest="early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--fc-drop-rate", dest="fc_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--drop-path", dest="drop_path", type=float, default=0.1)
    parser.add_argument("--attn-drop-rate", dest="attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--use-checkpoint", dest="use_checkpoint", action="store_true")
    parser.add_argument("--use-mean-pooling", dest="use_mean_pooling", action="store_true")
    parser.add_argument("--init-scale", dest="init_scale", type=float, default=0.001)
    parser.add_argument("--deterministic-spatial", dest="deterministic_spatial", action="store_true")
    parser.add_argument("--view-mode", choices=["random", "center_uniform"], default="random")
    parser.add_argument("--target-cache-root", type=Path, default=None)
    parser.add_argument("--precompute-target-cache", action="store_true")
    parser.add_argument("--precompute-target-cache-only", action="store_true")
    parser.add_argument("--predownsampled", action="store_true",
                        help="Pseudo masks are already downsampled to [C,T',H',W'] by build_pseudo_labels.py. "
                             "Skip frame selection, crop, resize, and pooling at load time.")
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--pos-weight-max", type=float, default=10.0)
    parser.add_argument("--pos-weight-eps", type=float, default=1e-6)
    parser.add_argument("--pos-weight-cache", type=Path, default=None)
    parser.set_defaults(use_checkpoint=False, use_mean_pooling=True)
    return parser.parse_args()


def _collate_batch(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    targets = torch.stack([item[2] for item in batch], dim=0)
    metas = [item[3] for item in batch]
    return inputs, labels, targets, metas


def _build_dataset(
    anno_path: Path,
    data_root: Path,
    cli_args: argparse.Namespace,
) -> LocalConceptVideoDataset:
    target_cache_root = None
    if cli_args.target_cache_root is not None:
        target_cache_root = cli_args.target_cache_root / anno_path.stem
    return LocalConceptVideoDataset(
        anno_path=anno_path,
        data_root=data_root,
        data_set=cli_args.data_set,
        pseudo_mask_root=cli_args.pseudo_mask_root,
        tubelet_size=cli_args.tubelet_size,
        patch_size=cli_args.patch_size,
        target_cache_root=target_cache_root,
        num_frames=cli_args.num_frames,
        sampling_rate=cli_args.sampling_rate,
        input_size=cli_args.input_size,
        deterministic=cli_args.deterministic_spatial,
        view_mode=cli_args.view_mode,
        predownsampled=cli_args.predownsampled,
    )


def _precompute_target_cache(
    anno_path: Path,
    data_root: Path,
    cli_args: argparse.Namespace,
) -> None:
    dataset = _build_dataset(anno_path=anno_path, data_root=data_root, cli_args=cli_args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli_args.num_workers,
        pin_memory=False,
        collate_fn=_collate_batch,
    )
    progress = tqdm(loader, desc=f"target-cache-{anno_path.stem}", dynamic_ncols=True, disable=not _is_main_process())
    for _inputs, _labels, _targets, _metas in progress:
        pass


@torch.no_grad()
def _compute_pos_weight(
    anno_path: Path,
    data_root: Path,
    cli_args: argparse.Namespace,
) -> torch.Tensor:
    dataset = _build_dataset(anno_path=anno_path, data_root=data_root, cli_args=cli_args)
    loader = DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=cli_args.num_workers,
        pin_memory=False,
        collate_fn=_collate_batch,
    )
    progress = tqdm(loader, desc=f"pos-weight-{anno_path.stem}", dynamic_ncols=True, disable=not _is_main_process())

    positive = torch.zeros(cli_args.num_concepts, dtype=torch.float64)
    total_cells = 0.0
    for _inputs, _labels, targets, _metas in progress:
        targets = targets.to(torch.float32)
        positive += targets.sum(dim=(0, 2, 3, 4), dtype=torch.float64)
        total_cells += float(targets.shape[0] * targets.shape[2] * targets.shape[3] * targets.shape[4])

    negative = total_cells - positive
    raw = torch.sqrt(negative / (positive + cli_args.pos_weight_eps))
    normalized = raw / raw.mean().clamp_min(cli_args.pos_weight_eps)
    pos_weight = torch.clamp(normalized, min=0.5, max=cli_args.pos_weight_max).to(torch.float32)
    return pos_weight


def _resolve_pos_weight(cli_args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    if not cli_args.use_pos_weight:
        return None

    cache_path = cli_args.pos_weight_cache or (cli_args.output_dir / "pos_weight.pt")
    pos_weight_cpu = None

    if _is_main_process():
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            if isinstance(cached, dict) and "pos_weight" in cached:
                pos_weight_cpu = cached["pos_weight"].to(torch.float32)
            elif isinstance(cached, torch.Tensor):
                pos_weight_cpu = cached.to(torch.float32)
            else:
                raise ValueError(f"Unsupported pos_weight cache format at {cache_path}")
        else:
            pos_weight_cpu = _compute_pos_weight(
                anno_path=cli_args.anno_path,
                data_root=cli_args.data_root,
                cli_args=cli_args,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "pos_weight": pos_weight_cpu,
                    "mode": "sqrt_neg_over_pos_mean_normalized",
                    "max": cli_args.pos_weight_max,
                    "eps": cli_args.pos_weight_eps,
                    "anno_path": str(cli_args.anno_path),
                },
                cache_path,
            )
            with open(cache_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": "sqrt_neg_over_pos_mean_normalized",
                        "max": cli_args.pos_weight_max,
                        "eps": cli_args.pos_weight_eps,
                        "values": [float(x) for x in pos_weight_cpu.tolist()],
                    },
                    f,
                    indent=2,
                )

    if _is_distributed():
        if pos_weight_cpu is None:
            pos_weight_cpu = torch.zeros(cli_args.num_concepts, dtype=torch.float32)
        pos_weight_device = pos_weight_cpu.to(device)
        dist.broadcast(pos_weight_device, src=0)
        return pos_weight_device

    assert pos_weight_cpu is not None
    return pos_weight_cpu.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.train()
    metrics = RunningBinaryLocalizationMetrics()

    progress = tqdm(loader, desc="train", leave=False, dynamic_ncols=True, disable=not _is_main_process())
    for inputs, _, targets, _ in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, _ = model(inputs)
        if logits.shape != targets.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}")

        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.update(logits=logits.detach(), targets=targets.detach(), loss=loss.detach(), threshold=threshold)
        current = metrics.compute()
        progress.set_postfix(loss=f"{current['loss']:.4f}", iou=f"{current['iou']:.4f}", recall=f"{current['recall']:.4f}")

    metrics.synchronize_between_processes(device)
    return metrics.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    threshold: float,
    preview_dir: Path | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    model.eval()
    metrics = RunningBinaryLocalizationMetrics()
    preview_saved = False

    progress = tqdm(loader, desc="val", leave=False, dynamic_ncols=True, disable=not _is_main_process())
    for inputs, _, targets, metas in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, _ = model(inputs)
        if logits.shape != targets.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}")

        loss = criterion(logits, targets)
        metrics.update(logits=logits, targets=targets, loss=loss, threshold=threshold)
        current = metrics.compute()
        progress.set_postfix(loss=f"{current['loss']:.4f}", iou=f"{current['iou']:.4f}", recall=f"{current['recall']:.4f}")

        if (
            preview_dir is not None
            and epoch is not None
            and not preview_saved
            and _is_main_process()
            and inputs.shape[1] == 3
        ):
            save_localization_previews(
                videos=inputs.detach().cpu(),
                logits=logits.detach().cpu(),
                targets=targets.detach().cpu(),
                metas=metas,
                output_dir=preview_dir,
                epoch=epoch,
                max_samples=args.preview_max_samples,
                threshold=threshold,
            )
            preview_saved = True

    metrics.synchronize_between_processes(device)
    return metrics.compute()


def _make_loader(
    anno_path: Path,
    data_root: Path,
    cli_args: argparse.Namespace,
    shuffle: bool,
    use_distributed_sampler: bool = True,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = _build_dataset(anno_path=anno_path, data_root=data_root, cli_args=cli_args)
    sampler = None
    if use_distributed_sampler and _is_distributed():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=cli_args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=cli_args.num_workers,
        pin_memory=True,
        collate_fn=_collate_batch,
    )
    return loader, sampler


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "head_type": _unwrap_model(model).head_type,
        },
        output_dir / name,
    )


def _infer_num_concepts(mask_root: Path, anno_path: Path) -> int:
    with open(anno_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Could not infer num concepts from empty annotation file: {anno_path}")

    for line in lines:
        sample = line.split(",", 1)[0].strip()
        sample_id = _build_sample_id(sample)
        mask_path = mask_root / sample_id / "pixel_mask.npy"
        if not mask_path.exists():
            continue
        mask = torch.from_numpy(np.load(mask_path))
        if mask.ndim == 3:
            return 1
        if mask.ndim == 4:
            return int(mask.shape[0])
        raise ValueError(f"Unsupported pixel mask shape at {mask_path}: {tuple(mask.shape)}")

    raise FileNotFoundError(
        f"Could not find any pixel_mask.npy under {mask_root} for samples listed in {anno_path}"
    )


def main() -> None:
    cli_args = parse_args()
    device, rank, world_size, local_rank = _setup_distributed(cli_args)
    try:
        if cli_args.precompute_target_cache_only:
            cli_args.precompute_target_cache = True
        if cli_args.num_concepts <= 0:
            cli_args.num_concepts = _infer_num_concepts(cli_args.pseudo_mask_root, cli_args.anno_path)
            if _is_main_process():
                print(f"Inferred num_concepts={cli_args.num_concepts} from pseudo masks under {cli_args.pseudo_mask_root}")
        if _is_main_process() and world_size > 1:
            print(f"Running distributed training with world_size={world_size}, local_rank={local_rank}")
        if _is_main_process():
            print(f"Using view_mode={cli_args.view_mode}")

        model_args = build_videomae_args(cli_args)
        output_dir = cli_args.output_dir
        if _is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)

        if cli_args.precompute_target_cache:
            if cli_args.target_cache_root is None:
                raise ValueError("--precompute-target-cache requires --target-cache-root.")
            if cli_args.view_mode != "center_uniform":
                raise ValueError("--precompute-target-cache is only supported with --view-mode center_uniform.")
            if _is_main_process():
                _precompute_target_cache(
                    anno_path=cli_args.anno_path,
                    data_root=cli_args.data_root,
                    cli_args=cli_args,
                )
                if cli_args.val_anno_path is not None:
                    _precompute_target_cache(
                        anno_path=cli_args.val_anno_path,
                        data_root=cli_args.val_data_root or cli_args.data_root,
                        cli_args=cli_args,
                    )
            if _is_distributed():
                dist.barrier()
            if cli_args.precompute_target_cache_only:
                return

        backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
        train_loader, train_sampler = _make_loader(
            anno_path=cli_args.anno_path,
            data_root=cli_args.data_root,
            cli_args=cli_args,
            shuffle=True,
        )
        val_loader = None
        val_sampler = None
        if cli_args.val_anno_path is not None:
            val_loader, val_sampler = _make_loader(
                anno_path=cli_args.val_anno_path,
                data_root=cli_args.val_data_root or cli_args.data_root,
                cli_args=cli_args,
                shuffle=False,
            )

        model = VideoMAELocalizer(
            backbone,
            out_channels=cli_args.num_concepts,
            head_type=cli_args.head_type,
            num_heads=cli_args.head_num_heads,
            head_dropout=cli_args.head_dropout,
        ).to(device)
        if _is_distributed():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        pos_weight = _resolve_pos_weight(cli_args, device)
        if pos_weight is not None:
            # BCEWithLogitsLoss broadcasts from the trailing dimensions, so a
            # flat [C] vector would incorrectly align to width for [B, C, T, H, W].
            pos_weight = pos_weight.view(1, cli_args.num_concepts, 1, 1, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(_unwrap_model(model).head.parameters(), lr=cli_args.lr)
        metrics_path = output_dir / "metrics.jsonl"
        best_iou = float("-inf")
        epochs_without_improvement = 0

        if _is_main_process() and pos_weight is not None:
            print(
                "Using pos_weight (sqrt(neg/pos), mean-normalized, clipped): "
                + ", ".join(f"{idx}:{value:.4f}" for idx, value in enumerate(pos_weight.detach().view(-1).cpu().tolist()))
            )

        for epoch in range(cli_args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                cli_args,
                device,
                threshold=cli_args.eval_threshold,
            )
            summary = {
                "epoch": epoch,
                "train": train_metrics,
            }

            if val_loader is not None:
                preview_dir = None
                if epoch % cli_args.save_preview_every == 0:
                    preview_dir = output_dir / "previews"
                val_metrics = evaluate(
                    model,
                    val_loader,
                    criterion,
                    cli_args,
                    device,
                    threshold=cli_args.eval_threshold,
                    preview_dir=preview_dir,
                    epoch=epoch,
                )
                summary["val"] = val_metrics
                current_iou = val_metrics["iou"]
            else:
                preview_dir = None
                if epoch % cli_args.save_preview_every == 0:
                    preview_dir = output_dir / "previews"
                val_metrics = evaluate(
                    model,
                    train_loader,
                    criterion,
                    cli_args,
                    device,
                    threshold=cli_args.eval_threshold,
                    preview_dir=preview_dir,
                    epoch=epoch,
                )
                summary["preview_split"] = "train"
                summary["preview_metrics"] = val_metrics
                current_iou = train_metrics["iou"]

            if _is_main_process():
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary) + "\n")

                print(
                    f"[epoch {epoch}] "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"train_iou={train_metrics['iou']:.4f} "
                    f"train_f1={train_metrics['f1']:.4f} "
                    f"train_recall={train_metrics['recall']:.4f} "
                    f"pred_pos={train_metrics['pred_positive_ratio']:.4f} "
                    f"target_pos={train_metrics['target_positive_ratio']:.4f}"
                )
                if "val" in summary:
                    print(
                        f"[epoch {epoch}] "
                        f"val_loss={summary['val']['loss']:.6f} "
                        f"val_iou={summary['val']['iou']:.4f} "
                        f"val_f1={summary['val']['f1']:.4f} "
                        f"val_recall={summary['val']['recall']:.4f}"
                    )

                _save_checkpoint(model, optimizer, epoch, summary, output_dir, "last.pt")
                if current_iou > best_iou + cli_args.early_stopping_min_delta:
                    best_iou = current_iou
                    epochs_without_improvement = 0
                    _save_checkpoint(model, optimizer, epoch, summary, output_dir, "best.pt")
                else:
                    epochs_without_improvement += 1

                if cli_args.early_stopping_patience > 0 and epochs_without_improvement >= cli_args.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch}: "
                        f"no improvement in monitored IoU for {cli_args.early_stopping_patience} epoch(s). "
                        f"best_iou={best_iou:.4f}"
                    )

            should_stop = False
            if _is_distributed():
                stop_tensor = torch.tensor(
                    1 if (_is_main_process() and cli_args.early_stopping_patience > 0 and epochs_without_improvement >= cli_args.early_stopping_patience) else 0,
                    device=device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            else:
                should_stop = cli_args.early_stopping_patience > 0 and epochs_without_improvement >= cli_args.early_stopping_patience
            if should_stop:
                break
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()

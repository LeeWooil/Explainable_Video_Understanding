"""Baseline CBM: reproduce PCBEAR pipeline within our framework.

Pipeline:
  Video -> Full VideoMAE forward_features() -> [B, D]
       -> W_c Linear(D, C) -> [B, C]  (concept scores)
       -> W_g (GLM-SAGA sparse classifier) -> [B, num_classes]

This is a sanity check to verify we can match PCBEAR's 90% val accuracy
using the same approach but within our codebase.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from configs.defaults import build_videomae_args
from datasets.local_video_dataset import LocalVideoDataset, _build_sample_id
from models.backbone import FrozenVideoMAEBackbone


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Baseline CBM: full VideoMAE forward -> Linear(D, C) -> GLM-SAGA."
    )
    # data
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--val-anno-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-data-root", type=Path, default=None)
    parser.add_argument("--global-label-dir", type=Path, required=True)
    parser.add_argument("--video-anno-path", type=Path, required=True)
    # model
    parser.add_argument("--backbone", type=str, default="vmae_vit_base_patch16_224")
    parser.add_argument("--finetune", type=str, required=True)
    parser.add_argument("--data-set", type=str, required=True)
    parser.add_argument("--nb-classes", dest="nb_classes", type=int, required=True)
    parser.add_argument("--num-concepts", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--block-index", type=int, default=11)
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
    # W_c training
    parser.add_argument("--proj-steps", type=int, default=3000)
    parser.add_argument("--proj-batch-size", type=int, default=50000)
    parser.add_argument("--proj-lr", type=float, default=1e-3)
    parser.add_argument("--use-mlp", action="store_true")
    parser.add_argument(
        "--loss-mode",
        default="concept",
        choices=["concept", "sample", "second", "first_concept", "first_sample"],
    )
    parser.add_argument("--no-filter-out", action="store_true")
    # W_g training
    parser.add_argument("--saga-batch-size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.0007)
    parser.add_argument("--n-iters", type=int, default=30000)
    # output
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(use_checkpoint=False, use_mean_pooling=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
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


def make_loader(
    anno_path: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> tuple[DataLoader, list[int]]:
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
    return loader, list(dataset.label_array)


def load_global_labels(label_path: Path) -> tuple[torch.Tensor, list[str]]:
    with open(label_path, "rb") as f:
        data = pickle.load(f)
    video_names = [str(item["video_name"]) for item in data]
    labels = torch.tensor(
        [item["attribute_label"] for item in data], dtype=torch.float32
    )
    return labels, video_names


def normalize_name(name: str) -> str:
    return _build_sample_id(Path(name).name)


def validate_alignment(
    sample_ids: list[str], video_names: list[str], split_name: str
) -> None:
    if len(sample_ids) != len(video_names):
        raise ValueError(
            f"{split_name}: feature count {len(sample_ids)} does not match "
            f"global label count {len(video_names)}"
        )
    mismatches = []
    for idx, (sid, vname) in enumerate(zip(sample_ids, video_names)):
        if normalize_name(vname) != sid:
            mismatches.append((idx, sid, vname))
            if len(mismatches) >= 5:
                break
    if mismatches:
        details = "; ".join(
            f"{i}: {s} != {v}" for i, s, v in mismatches
        )
        raise ValueError(
            f"{split_name}: sample order mismatch between video loader "
            f"and global labels: {details}"
        )


def select_similarity_fn(loss_mode: str):
    from utils import similarity

    lookup = {
        "concept": similarity.cos_similarity_cubed_single_concept,
        "sample": similarity.cos_similarity_cubed_single_sample,
        "second": similarity.cos_similarity_cubed_single_secondpower,
        "first_concept": similarity.cos_similarity_cubed_single_firstpower_concept,
        "first_sample": similarity.cos_similarity_cubed_single_firstpower_sample,
    }
    if loss_mode not in lookup:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    return lookup[loss_mode]


# ---------------------------------------------------------------------------
# Feature extraction: full VideoMAE forward_features()
# ---------------------------------------------------------------------------

def extract_global_features(
    loader: DataLoader,
    vmae_model: nn.Module,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract global features via model.forward_features().

    This exactly replicates PCBEAR's feature extraction:
    all blocks -> norm -> mean pool -> fc_norm -> [B, D]

    Returns:
        features: [N, D]
        labels: [N]
        sample_ids: list of length N
    """
    feat_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []

    progress = tqdm(loader, desc=desc, dynamic_ncols=True)
    with torch.no_grad():
        for videos, labels, metas in progress:
            videos = videos.to(device, non_blocking=True)
            features = vmae_model.forward_features(videos)  # [B, D]
            feat_batches.append(features.cpu())
            label_batches.append(labels.cpu())
            sample_ids.extend(str(meta["sample_id"]) for meta in metas)

    all_features = torch.cat(feat_batches, dim=0)
    all_labels = torch.cat(label_batches, dim=0)

    feat_norm = all_features.norm(dim=1)
    feat_std = all_features.std(dim=0)
    print(f"[{desc}] Features: {tuple(all_features.shape)}")
    print(f"  L2 norm: mean={feat_norm.mean():.2f}, std={feat_norm.std():.2f}")
    print(f"  per-dim std: mean={feat_std.mean():.4f}, "
          f"min={feat_std.min():.4f}, max={feat_std.max():.4f}")

    return all_features, all_labels, sample_ids


# ---------------------------------------------------------------------------
# W_c: Linear(D, C)  [N, D] -> [N, C]
# ---------------------------------------------------------------------------

def train_concept_projection(
    args: argparse.Namespace,
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    save_dir: Path,
) -> tuple[torch.Tensor, float]:
    """Train W_c: Linear(D, C)."""
    embed_dim = train_features.shape[1]
    num_concepts = train_labels.shape[1]

    train_targets = train_labels.clone()
    val_targets = val_labels.clone()
    if not args.use_mlp:
        train_targets[train_targets == 0.0] = 0.05
        train_targets[train_targets == 1.0] = 0.3
        val_targets[val_targets == 0.0] = 0.05
        val_targets[val_targets == 1.0] = 0.3

    if args.no_filter_out:
        train_targets[train_targets == -1.0] = 1e-8
        val_targets[val_targets == -1.0] = 1e-8

    train_valid = torch.where(train_targets.max(dim=1).values != -1)[0]
    val_valid = torch.where(val_targets.max(dim=1).values != -1)[0]
    train_feat_valid = train_features[train_valid]
    train_tgt_valid = train_targets[train_valid]
    val_feat_valid = val_features[val_valid]
    val_tgt_valid = val_targets[val_valid]

    proj_layer = nn.Linear(embed_dim, num_concepts, bias=False).to(args.device)
    if args.use_mlp:
        nn.init.xavier_uniform_(proj_layer.weight)
        criterion = nn.BCEWithLogitsLoss()
    similarity_fn = None if args.use_mlp else select_similarity_fn(args.loss_mode)
    optimizer = torch.optim.Adam(proj_layer.parameters(), lr=args.proj_lr)

    print(f"W_c params: Linear({embed_dim}, {num_concepts}), "
          f"total={embed_dim * num_concepts}")

    indices = list(range(len(train_feat_valid)))
    proj_batch_size = min(args.proj_batch_size, len(train_feat_valid))
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None

    for step in range(args.proj_steps):
        batch_idx = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        batch_feat = train_feat_valid[batch_idx].to(args.device).detach()
        batch_tgt = train_tgt_valid[batch_idx].to(args.device).detach()

        outputs = proj_layer(batch_feat)

        if args.use_mlp:
            loss = criterion(outputs, batch_tgt)
        else:
            loss = -similarity_fn(batch_tgt, outputs)
        loss = torch.mean(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0 or step == args.proj_steps - 1:
            with torch.no_grad():
                val_out = proj_layer(val_feat_valid.to(args.device))
                if args.use_mlp:
                    val_loss = criterion(val_out, val_tgt_valid.to(args.device))
                else:
                    val_loss = -similarity_fn(val_tgt_valid.to(args.device), val_out)
                val_loss = torch.mean(val_loss)

            if step == 0 or val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_weights = proj_layer.weight.clone().detach().cpu()

            score = val_loss.item() if args.use_mlp else -val_loss.item()
            train_score = loss.item() if args.use_mlp else -loss.item()
            print(
                f"Step:{step}, Avg train similarity:{train_score:.4f}, "
                f"Avg val similarity:{score:.4f}"
            )

    assert best_weights is not None
    print(
        f"Best step:{best_step}, Avg val similarity:"
        f"{(-best_val_loss if not args.use_mlp else best_val_loss).item():.4f}"
    )

    proj_layer.load_state_dict({"weight": best_weights})
    proj_layer = proj_layer.cpu()

    with torch.no_grad():
        train_c = proj_layer(train_features.detach())
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
    torch.save(train_mean, save_dir / "proj_mean.pt")
    torch.save(train_std, save_dir / "proj_std.pt")
    w_c = proj_layer.weight[:]
    torch.save(w_c, save_dir / "W_c.pt")
    return w_c, float(best_val_loss.item())


# ---------------------------------------------------------------------------
# W_g: sparse classifier via GLM-SAGA
# ---------------------------------------------------------------------------

def train_classifier(
    args: argparse.Namespace,
    w_c: torch.Tensor,
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    save_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    proj_layer = nn.Linear(w_c.shape[1], w_c.shape[0], bias=False)
    proj_layer.load_state_dict({"weight": w_c})

    with torch.no_grad():
        train_c = proj_layer(train_features.detach())
        val_c = proj_layer(val_features.detach())

        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True).clamp_min(1e-6)

        train_c = (train_c - train_mean) / train_std
        val_c = (val_c - train_mean) / train_std

    torch.save(train_mean, save_dir / "proj_mean.pt")
    torch.save(train_std, save_dir / "proj_std.pt")

    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c, val_y)
    indexed_train_loader = DataLoader(
        indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.saga_batch_size, shuffle=False
    )

    cls_file = args.video_anno_path / "class_list.txt"
    with open(cls_file, "r", encoding="utf-8") as f:
        classes = f.read().split("\n")
    assert args.nb_classes == len(classes)

    linear = nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        0.05,
        args.n_iters,
        0.99,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata={"max_reg": {"nongrouped": args.lam}},
        n_ex=len(train_features),
        n_classes=len(classes),
        verbose=500,
    )
    w_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]

    torch.save(w_g, save_dir / "W_g.pt")
    torch.save(b_g, save_dir / "b_g.pt")
    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        out_dict = {}
        for key in ("lam", "lr", "alpha", "time"):
            out_dict[key] = float(output_proj["path"][0][key])
        out_dict["metrics"] = output_proj["path"][0]["metrics"]
        nnz = (w_g.abs() > 1e-5).sum().item()
        total = w_g.numel()
        out_dict["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total,
        }
        json.dump(out_dict, f, indent=2)
    return train_c, val_c


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    save_dir = args.save_dir / (
        f"{args.data_set}_baseline_cbm"
        f"_{args.num_concepts}concepts"
        f"_{timestamp}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            f,
            indent=2,
        )

    # -- Data loaders --
    train_loader, train_labels = make_loader(args.anno_path, args.data_root, args)
    val_loader, val_labels = make_loader(
        args.val_anno_path, args.val_data_root or args.data_root, args
    )

    # -- Load VideoMAE model (full, not frozen backbone) --
    model_args = build_videomae_args(args)
    backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
    vmae_model = backbone.model  # the raw VisionTransformer
    vmae_model.eval()
    for param in vmae_model.parameters():
        param.requires_grad = False
    print(f"VideoMAE model loaded. fc_norm: {vmae_model.fc_norm}")
    print(f"use_mean_pooling: {hasattr(vmae_model, 'fc_norm') and vmae_model.fc_norm is not None}")

    # -- Extract global features [N, D] via forward_features() --
    print("Extracting global features via forward_features()...")
    train_features, train_y, train_sample_ids = extract_global_features(
        loader=train_loader,
        vmae_model=vmae_model,
        device=device,
        desc="extract-train",
    )
    val_features, val_y, val_sample_ids = extract_global_features(
        loader=val_loader,
        vmae_model=vmae_model,
        device=device,
        desc="extract-val",
    )

    if train_labels != train_y.tolist():
        raise ValueError("Train label order mismatch.")
    if val_labels != val_y.tolist():
        raise ValueError("Val label order mismatch.")

    torch.save(train_features, save_dir / "train_features.pt")
    torch.save(val_features, save_dir / "val_features.pt")

    # -- Load global concept labels --
    train_global_labels, train_video_names = load_global_labels(
        args.global_label_dir / "hard_label_train.pkl"
    )
    val_global_labels, val_video_names = load_global_labels(
        args.global_label_dir / "hard_label_val.pkl"
    )
    validate_alignment(train_sample_ids, train_video_names, "train")
    validate_alignment(val_sample_ids, val_video_names, "val")
    if train_global_labels.shape[1] != args.num_concepts:
        raise ValueError(
            f"Global label dim {train_global_labels.shape[1]} does not match "
            f"--num-concepts {args.num_concepts}"
        )

    concepts = [str(i) for i in range(train_global_labels.shape[1])]
    with open(save_dir / "concepts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(concepts))

    # -- Train W_c: Linear(D, C) --
    print("Training concept projection (W_c)...")
    w_c, best_val_loss = train_concept_projection(
        args=args,
        train_features=train_features,
        val_features=val_features,
        train_labels=train_global_labels,
        val_labels=val_global_labels,
        save_dir=save_dir,
    )
    print(f"Concept layer best val loss: {best_val_loss:.6f}")

    # -- Train W_g: sparse classifier via GLM-SAGA --
    print("Training sparse classifier (W_g)...")
    train_c, val_c = train_classifier(
        args=args,
        w_c=w_c,
        train_features=train_features,
        val_features=val_features,
        train_y=train_y,
        val_y=val_y,
        save_dir=save_dir,
    )
    torch.save(train_c, save_dir / "train_c.pt")
    torch.save(val_c, save_dir / "val_c.pt")


if __name__ == "__main__":
    main()

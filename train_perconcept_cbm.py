"""Train a global CBM using per-concept attention pooling.

Pipeline:
  Video -> FrozenLocalizer -> (concept_logits [B,C,T,H,W], feature_map [B,D,T,H,W])
       -> ConceptGuidedAttentionPool -> [B, C, D]  (per-concept attended features)
       -> fc_norm (LayerNorm)
       -> W_c [C, D] per-concept projection: score[c] = W_c[c] · feat[c] -> [B, C]
       -> W_g (GLM-SAGA sparse classifier) -> [B, num_classes]

Key difference from train_attn_global_cbm.py:
  - Each concept pools its OWN spatial region from the backbone features
  - Concept scores come from concept-specific features, not a shared global feature
  - W_c[c] is a per-concept weight vector [D], each row operates on its own attended feature
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
from models.attention_pool import ConceptGuidedAttentionPool
from models.backbone import FrozenVideoMAEBackbone
from models.localizer import VideoMAELocalizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train a global CBM with per-concept attention pooling."
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
    # W_c training
    parser.add_argument("--proj-steps", type=int, default=3000)
    parser.add_argument("--proj-batch-size", type=int, default=50000)
    parser.add_argument("--proj-lr", type=float, default=1e-3)
    parser.add_argument("--attention-temperature", type=float, default=5.0,
                        help="Temperature for softmax attention (higher = smoother)")
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


# ---------------------------------------------------------------------------
# Feature extraction with per-concept attention pooling
# ---------------------------------------------------------------------------

def extract_perconcept_features(
    loader: DataLoader,
    model: VideoMAELocalizer,
    pool: ConceptGuidedAttentionPool,
    fc_norm: nn.Module,
    device: torch.device,
    desc: str,
    temperature: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract per-concept attended features.

    Returns:
        pooled_features: [N, C, D]  (per-concept features)
        labels: [N]
        sample_ids: list of length N
    """
    pooled_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    n_batches = 0

    progress = tqdm(loader, desc=desc, dynamic_ncols=True)
    with torch.no_grad():
        for videos, labels, metas in progress:
            videos = videos.to(device, non_blocking=True)
            concept_logits, feature_map = model(videos)
            # concept_logits: [B, C, T, H, W]
            # feature_map:    [B, D, T, H, W]

            # Per-concept attention pooling -> [B, C, D]
            pooled = pool(feature_map, concept_logits)

            # Apply LayerNorm per concept (over D dimension)
            B, C, D = pooled.shape
            pooled_flat = pooled.view(B * C, D)
            pooled_flat = fc_norm(pooled_flat)
            pooled = pooled_flat.view(B, C, D)

            pooled_batches.append(pooled.cpu())
            label_batches.append(labels.cpu())
            sample_ids.extend(str(meta["sample_id"]) for meta in metas)

            if n_batches == 0:
                print(f"[{desc}] concept_logits: {tuple(concept_logits.shape)}")
                print(f"[{desc}] feature_map: {tuple(feature_map.shape)}")
                print(f"[{desc}] pooled: {tuple(pooled.shape)}")

                # Attention diagnostics
                logit_flat = concept_logits.view(B, C, -1)
                attn = torch.softmax(logit_flat / temperature, dim=-1)
                S = logit_flat.shape[-1]
                uniform = 1.0 / S
                print(f"[{desc}] Attention stats (temp={temperature}):")
                print(f"  spatial std (avg): {attn.std(dim=-1).mean():.6f}")
                print(f"  max weight (avg): {attn.max(dim=-1).values.mean():.6f}")
                print(f"  uniform baseline: {uniform:.6f}")
            n_batches += 1

    return (
        torch.cat(pooled_batches, dim=0),
        torch.cat(label_batches, dim=0),
        sample_ids,
    )


# ---------------------------------------------------------------------------
# W_c: per-concept projection  W_c[c] · feat[c] -> score[c]
# W_c shape: [C, D],  features shape: [N, C, D]  ->  scores: [N, C]
# ---------------------------------------------------------------------------

def _perconcept_forward(weight: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Apply per-concept projection: score[b,c] = W_c[c] · feat[b,c].

    Args:
        weight: [C, D]
        features: [B, C, D]
    Returns:
        scores: [B, C]
    """
    return torch.einsum("cd, bcd -> bc", weight, features)


def train_concept_projection(
    args: argparse.Namespace,
    train_features: torch.Tensor,   # [N, C, D]
    val_features: torch.Tensor,     # [N, C, D]
    train_labels: torch.Tensor,     # [N, C]
    val_labels: torch.Tensor,       # [N, C]
    save_dir: Path,
) -> tuple[torch.Tensor, float]:
    """Train W_c with per-concept independent projections.

    Each concept c has its own weight vector W_c[c] of shape [D].
    score[b, c] = W_c[c] · features[b, c]

    Returns:
        best_weights: [C, D] weight matrix
        best_val_loss: float
    """
    N_train, C, D = train_features.shape

    # Filter valid samples (not all -1 in global labels)
    train_valid = torch.where(train_labels.max(dim=1).values != -1)[0]
    val_valid = torch.where(val_labels.max(dim=1).values != -1)[0]
    train_feat_valid = train_features[train_valid]   # [N', C, D]
    train_tgt_valid = train_labels[train_valid]      # [N', C]
    val_feat_valid = val_features[val_valid]
    val_tgt_valid = val_labels[val_valid]

    # W_c: [C, D] — each row is a per-concept projection vector
    w_c = nn.Parameter(torch.empty(C, D, device=args.device))
    nn.init.xavier_uniform_(w_c)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([w_c], lr=args.proj_lr)

    print(f"W_c params: per-concept projection [{C}, {D}], "
          f"total={C * D} trainable params")

    indices = list(range(len(train_feat_valid)))
    proj_batch_size = min(args.proj_batch_size, len(train_feat_valid))
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None

    for step in range(args.proj_steps):
        batch_idx = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        batch_feat = train_feat_valid[batch_idx].to(args.device)   # [B, C, D]
        batch_tgt = train_tgt_valid[batch_idx].to(args.device)     # [B, C]

        outputs = _perconcept_forward(w_c, batch_feat)             # [B, C]

        loss = criterion(outputs, batch_tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0 or step == args.proj_steps - 1:
            with torch.no_grad():
                val_out = _perconcept_forward(w_c, val_feat_valid.to(args.device))
                val_loss = criterion(val_out, val_tgt_valid.to(args.device))

            if step == 0 or val_loss <= best_val_loss:
                best_val_loss = val_loss.item()
                best_step = step
                best_weights = w_c.clone().detach().cpu()

            print(
                f"Step:{step}, Train loss:{loss.item():.4f}, "
                f"Val loss:{val_loss.item():.4f}"
            )

    assert best_weights is not None
    print(f"Best step:{best_step}, Best val loss:{best_val_loss:.4f}")

    # Compute concept scores for normalization stats
    with torch.no_grad():
        train_c = _perconcept_forward(best_weights, train_features)  # [N, C]
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)

    torch.save(train_mean, save_dir / "proj_mean.pt")
    torch.save(train_std, save_dir / "proj_std.pt")
    torch.save(best_weights, save_dir / "W_c.pt")
    return best_weights, best_val_loss


# ---------------------------------------------------------------------------
# W_g: sparse classifier via GLM-SAGA
# ---------------------------------------------------------------------------

def train_classifier(
    args: argparse.Namespace,
    w_c: torch.Tensor,              # [C, D]
    train_features: torch.Tensor,   # [N, C, D]
    val_features: torch.Tensor,     # [N, C, D]
    train_y: torch.Tensor,          # [N]
    val_y: torch.Tensor,            # [N]
    save_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project features through W_c, normalize, then train sparse W_g."""
    with torch.no_grad():
        # [N, C, D] × [C, D] -> [N, C]
        train_c = _perconcept_forward(w_c, train_features)
        val_c = _perconcept_forward(w_c, val_features)

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
    assert args.nb_classes == len(classes), (
        f"args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"
    )

    num_concepts = train_c.shape[1]
    linear = nn.Linear(num_concepts, len(classes)).to(args.device)
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
    temp_str = f"_temp{args.attention_temperature}" if args.attention_temperature != 1.0 else ""
    save_dir = args.save_dir / (
        f"{args.data_set}_perconcept_cbm"
        f"_block{args.block_index}"
        f"_{args.num_concepts}concepts"
        f"{temp_str}"
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

    # -- Load frozen localizer (backbone + concept head) --
    model_args = build_videomae_args(args)
    backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
    model = VideoMAELocalizer(backbone, out_channels=args.num_concepts).to(device)
    checkpoint = torch.load(args.localizer_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    pool = ConceptGuidedAttentionPool(temperature=args.attention_temperature)

    # -- Extract fc_norm from the backbone (LayerNorm applied after pooling) --
    vmae_model = model.backbone.model
    fc_norm = vmae_model.fc_norm if vmae_model.fc_norm is not None else nn.Identity()
    fc_norm = fc_norm.to(device)
    fc_norm.eval()
    print(f"fc_norm: {fc_norm}")

    # -- Extract per-concept attended features [N, C, D] --
    print("Extracting per-concept attended features...")
    train_features, train_y, train_sample_ids = extract_perconcept_features(
        loader=train_loader,
        model=model,
        pool=pool,
        fc_norm=fc_norm,
        device=device,
        desc="extract-train",
        temperature=args.attention_temperature,
    )
    val_features, val_y, val_sample_ids = extract_perconcept_features(
        loader=val_loader,
        model=model,
        pool=pool,
        fc_norm=fc_norm,
        device=device,
        desc="extract-val",
        temperature=args.attention_temperature,
    )

    if train_labels != train_y.tolist():
        raise ValueError("Train label order mismatch.")
    if val_labels != val_y.tolist():
        raise ValueError("Val label order mismatch.")

    print(f"Train features: {tuple(train_features.shape)}")  # [N, C, D]
    print(f"Val features:   {tuple(val_features.shape)}")

    # -- Feature distribution diagnostics --
    # Per-concept feature norms
    feat_norm = train_features.norm(dim=2)  # [N, C]
    print(f"Per-concept feature L2 norm: mean={feat_norm.mean():.2f}, std={feat_norm.std():.2f}")
    print(f"Per-concept norm variation across concepts: {feat_norm.mean(dim=0).std():.4f}")
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

    # -- Train W_c: Linear(D, 1) shared across concepts --
    print("Training per-concept projection (W_c)...")
    w_c, best_val_loss = train_concept_projection(
        args=args,
        train_features=train_features,
        val_features=val_features,
        train_labels=train_global_labels,
        val_labels=val_global_labels,
        save_dir=save_dir,
    )
    print(f"Concept projection best val loss: {best_val_loss:.6f}")

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

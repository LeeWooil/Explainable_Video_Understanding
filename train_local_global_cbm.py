from __future__ import annotations

import argparse
import json
import os
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
from models.localizer import VideoMAELocalizer
from utils import similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train a global concept layer and classifier on top of a frozen localizer."
    )
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--val-anno-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-data-root", type=Path, default=None)
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
    parser.add_argument("--pooling", choices=["mean", "max", "mean_max", "flatten"], default="mean")
    parser.add_argument("--pool-source", choices=["prob", "logit"], default="prob")
    parser.add_argument("--global-label-dir", type=Path, required=True)
    parser.add_argument("--video-anno-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proj-steps", type=int, default=1000)
    parser.add_argument("--proj-batch-size", type=int, default=50000)
    parser.add_argument("--use-mlp", action="store_true")
    parser.add_argument(
        "--loss-mode",
        default="concept",
        choices=["concept", "sample", "second", "first_concept", "first_sample"],
    )
    parser.add_argument("--no-filter-out", action="store_true")
    parser.add_argument("--saga-batch-size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.0007)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--fusion-mode", choices=["local", "concat", "gated", "learnable_gated"], default="local")
    parser.add_argument("--fusion-gate", type=float, default=0.5)
    parser.add_argument("--gate-steps", type=int, default=1000)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--gate-weight-decay", type=float, default=0.0)
    parser.add_argument("--global-pose-dir", type=Path, default=None)
    parser.add_argument("--global-backbone-train", type=Path, default=None)
    parser.add_argument("--global-backbone-val", type=Path, default=None)
    parser.set_defaults(use_checkpoint=False, use_mean_pooling=True)
    return parser.parse_args()


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


def pool_local_maps(logits: torch.Tensor, pooling: str, source: str) -> torch.Tensor:
    maps = torch.sigmoid(logits) if source == "prob" else logits
    if pooling == "mean":
        return maps.mean(dim=(2, 3, 4))
    if pooling == "max":
        return maps.amax(dim=(2, 3, 4))
    if pooling == "mean_max":
        pooled_mean = maps.mean(dim=(2, 3, 4))
        pooled_max = maps.amax(dim=(2, 3, 4))
        return torch.cat([pooled_mean, pooled_max], dim=1)
    if pooling == "flatten":
        # [B, C, T, H, W] -> [B, C*T*H*W]
        B = maps.shape[0]
        return maps.reshape(B, -1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def load_global_labels(label_path: Path) -> tuple[torch.Tensor, list[str]]:
    with open(label_path, "rb") as f:
        data = pickle.load(f)
    video_names = [str(item["video_name"]) for item in data]
    labels = torch.tensor([item["attribute_label"] for item in data], dtype=torch.float32)
    return labels, video_names


def load_concept_names(concepts_path: Path) -> list[str]:
    with open(concepts_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_name(name: str) -> str:
    return _build_sample_id(Path(name).name)


def validate_alignment(sample_ids: list[str], video_names: list[str], split_name: str) -> None:
    if len(sample_ids) != len(video_names):
        raise ValueError(
            f"{split_name}: feature count {len(sample_ids)} does not match global label count {len(video_names)}"
        )
    mismatches = []
    for idx, (sample_id, video_name) in enumerate(zip(sample_ids, video_names)):
        if normalize_name(video_name) != sample_id:
            mismatches.append((idx, sample_id, video_name))
            if len(mismatches) >= 5:
                break
    if mismatches:
        details = "; ".join(f"{idx}: {sample_id} != {video_name}" for idx, sample_id, video_name in mismatches)
        raise ValueError(f"{split_name}: sample order mismatch between video loader and global labels: {details}")


def extract_pooled_features(
    loader: DataLoader,
    model: VideoMAELocalizer,
    device: torch.device,
    pooling: str,
    pool_source: str,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    pooled_batches = []
    label_batches = []
    sample_ids: list[str] = []
    progress = tqdm(loader, desc=desc, dynamic_ncols=True)
    with torch.no_grad():
        for videos, labels, metas in progress:
            videos = videos.to(device, non_blocking=True)
            logits, _ = model(videos)
            pooled = pool_local_maps(logits, pooling, pool_source)
            pooled_batches.append(pooled.cpu())
            label_batches.append(labels.cpu())
            sample_ids.extend(str(meta["sample_id"]) for meta in metas)
    return torch.cat(pooled_batches, dim=0), torch.cat(label_batches, dim=0), sample_ids


def normalize_concept_features(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    save_dir: Path,
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_mean = torch.mean(train_features, dim=0, keepdim=True)
    train_std = torch.std(train_features, dim=0, keepdim=True).clamp_min(1e-6)
    train_norm = (train_features - train_mean) / train_std
    val_norm = (val_features - train_mean) / train_std
    torch.save(train_mean, save_dir / f"{prefix}_proj_mean.pt")
    torch.save(train_std, save_dir / f"{prefix}_proj_std.pt")
    return train_norm, val_norm


def build_global_pose_features(
    global_pose_dir: Path,
    train_backbone_features: torch.Tensor,
    val_backbone_features: torch.Tensor,
    save_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    w_c = torch.load(global_pose_dir / "W_c.pt", map_location="cpu").float()
    proj_mean = torch.load(global_pose_dir / "proj_mean.pt", map_location="cpu").float()
    proj_std = torch.load(global_pose_dir / "proj_std.pt", map_location="cpu").float().clamp_min(1e-6)
    proj_layer = nn.Linear(train_backbone_features.shape[1], w_c.shape[0], bias=False)
    proj_layer.load_state_dict({"weight": w_c})
    with torch.no_grad():
        train_c = proj_layer(train_backbone_features)
        val_c = proj_layer(val_backbone_features)
        train_c = (train_c - proj_mean) / proj_std
        val_c = (val_c - proj_mean) / proj_std
    torch.save(proj_mean, save_dir / "global_proj_mean.pt")
    torch.save(proj_std, save_dir / "global_proj_std.pt")
    concepts = load_concept_names(global_pose_dir / "concepts.txt")
    if len(concepts) != train_c.shape[1]:
        raise ValueError(
            f"Global concept count mismatch: concepts.txt has {len(concepts)} names but features have dim {train_c.shape[1]}"
        )
    return train_c, val_c, concepts


def select_similarity_fn(loss_mode: str):
    if loss_mode == "concept":
        return similarity.cos_similarity_cubed_single_concept
    if loss_mode == "sample":
        return similarity.cos_similarity_cubed_single_sample
    if loss_mode == "second":
        return similarity.cos_similarity_cubed_single_secondpower
    if loss_mode == "first_concept":
        return similarity.cos_similarity_cubed_single_firstpower_concept
    if loss_mode == "first_sample":
        return similarity.cos_similarity_cubed_single_firstpower_sample
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def train_global_concept_layer(
    args: argparse.Namespace,
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    save_dir: Path,
) -> tuple[torch.Tensor, float]:
    similarity_fn = select_similarity_fn(args.loss_mode)

    if not args.use_mlp:
        train_targets = train_labels.clone()
        val_targets = val_labels.clone()
        train_targets[train_targets == 0.0] = 0.05
        train_targets[train_targets == 1.0] = 0.3
        val_targets[val_targets == 0.0] = 0.05
        val_targets[val_targets == 1.0] = 0.3
    else:
        train_targets = train_labels.clone()
        val_targets = val_labels.clone()

    if args.no_filter_out:
        train_targets[train_targets == -1.0] = 1e-8
        val_targets[val_targets == -1.0] = 1e-8

    train_valid_index = torch.where(train_targets.max(dim=1).values != -1)[0]
    val_valid_index = torch.where(val_targets.max(dim=1).values != -1)[0]
    train_features_indexed = train_features[train_valid_index]
    train_targets_indexed = train_targets[train_valid_index]
    val_features_indexed = val_features[val_valid_index]
    val_targets_indexed = val_targets[val_valid_index]

    proj_layer = nn.Linear(
        in_features=train_features_indexed.shape[1],
        out_features=train_targets_indexed.shape[1],
        bias=False,
    ).to(args.device)
    if args.use_mlp:
        nn.init.xavier_uniform_(proj_layer.weight)
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = list(range(len(train_features_indexed)))
    proj_batch_size = min(args.proj_batch_size, len(train_features_indexed))
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None

    for step in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outputs = proj_layer(train_features_indexed[batch].to(args.device).detach())
        if args.use_mlp:
            loss = criterion(outputs, train_targets_indexed[batch].to(args.device).detach())
        else:
            loss = -select_similarity_fn(args.loss_mode)(
                train_targets_indexed[batch].to(args.device).detach(),
                outputs,
            )
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == args.proj_steps - 1:
            with torch.no_grad():
                val_output = proj_layer(val_features_indexed.to(args.device).detach())
                if args.use_mlp:
                    val_loss = criterion(val_output, val_targets_indexed.to(args.device).detach())
                else:
                    val_loss = -similarity_fn(val_targets_indexed.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if step == 0 or val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_weights = proj_layer.weight.clone()
            score = val_loss.cpu().item() if args.use_mlp else (-val_loss).cpu().item()
            train_score = loss.cpu().item() if args.use_mlp else (-loss).cpu().item()
            print(f"Step:{step}, Avg train similarity:{train_score:.4f}, Avg val similarity:{score:.4f}")
        optimizer.zero_grad()

    assert best_weights is not None
    proj_layer.load_state_dict({"weight": best_weights})
    print(f"Best step:{best_step}, Avg val similarity:{(-best_val_loss if not args.use_mlp else best_val_loss).cpu().item():.4f}")

    with torch.no_grad():
        proj_layer = proj_layer.cpu()
        train_c = proj_layer(train_features.detach())
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
    torch.save(train_mean, save_dir / "proj_mean.pt")
    torch.save(train_std, save_dir / "proj_std.pt")
    w_c = proj_layer.weight[:]
    torch.save(w_c, save_dir / "W_c.pt")
    return w_c, float(best_val_loss.cpu().item())


def train_classifier_identical(
    args: argparse.Namespace,
    w_c: torch.Tensor,
    concepts: list[str],
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    save_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    proj_layer = nn.Linear(train_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight": w_c})
    with torch.no_grad():
        train_c = proj_layer(train_features.detach())
        val_c = proj_layer(val_features.detach())

        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True).clamp_min(1e-6)

        train_c -= train_mean
        train_c /= train_std
        val_c -= train_mean
        val_c /= train_std

    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c, val_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    cls_file = args.video_anno_path / "class_list.txt"
    with open(cls_file, "r", encoding="utf-8") as f:
        classes = f.read().split("\n")
    assert args.nb_classes == len(classes), f"args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"

    linear = nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    step_size = 0.05
    alpha = 0.99
    metadata = {"max_reg": {"nongrouped": args.lam}}

    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        step_size,
        args.n_iters,
        alpha,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_features),
        n_classes=len(classes),
        verbose=500,
    )
    w_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]

    torch.save(train_mean, save_dir / "proj_mean.pt")
    torch.save(train_std, save_dir / "proj_std.pt")
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


def train_classifier_on_concepts(
    args: argparse.Namespace,
    train_c: torch.Tensor,
    val_c: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    save_dir: Path,
) -> None:
    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c, val_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    cls_file = args.video_anno_path / "class_list.txt"
    with open(cls_file, "r", encoding="utf-8") as f:
        classes = f.read().split("\n")
    assert args.nb_classes == len(classes), f"args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"

    linear = nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    step_size = 0.05
    alpha = 0.99
    metadata = {"max_reg": {"nongrouped": args.lam}}

    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        step_size,
        args.n_iters,
        alpha,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_c),
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


def train_learnable_gate(
    args: argparse.Namespace,
    train_local: torch.Tensor,
    val_local: torch.Tensor,
    train_global: torch.Tensor,
    val_global: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    save_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_concepts = train_local.shape[1]
    gate_logits = nn.Parameter(torch.zeros(num_concepts, device=args.device))
    cls_layer = nn.Linear(num_concepts, args.nb_classes).to(args.device)
    optimizer = torch.optim.Adam(
        [gate_logits, *cls_layer.parameters()],
        lr=args.gate_lr,
        weight_decay=args.gate_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    train_local = train_local.to(args.device)
    val_local = val_local.to(args.device)
    train_global = train_global.to(args.device)
    val_global = val_global.to(args.device)
    train_y = train_y.to(args.device)
    val_y = val_y.to(args.device)

    best_state = None
    best_val_acc = float("-inf")

    for step in range(args.gate_steps):
        gate = torch.sigmoid(gate_logits)
        fused_train = gate * train_local + (1.0 - gate) * train_global
        logits = cls_layer(fused_train)
        loss = criterion(logits, train_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == args.gate_steps - 1:
            with torch.no_grad():
                gate_eval = torch.sigmoid(gate_logits)
                fused_val = gate_eval * val_local + (1.0 - gate_eval) * val_global
                val_logits = cls_layer(fused_val)
                val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
                train_acc = (logits.argmax(dim=1) == train_y).float().mean().item()
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "gate": gate_eval.detach().cpu(),
                    "classifier": {k: v.detach().cpu() for k, v in cls_layer.state_dict().items()},
                    "step": step,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                }
            print(
                f"[gate step {step}] loss={loss.item():.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Failed to learn gate weights.")

    gate = best_state["gate"]
    torch.save(gate, save_dir / "learned_gate.pt")
    with open(save_dir / "fusion_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fusion_mode": "learnable_gated",
                "best_step": best_state["step"],
                "best_train_acc": best_state["train_acc"],
                "best_val_acc": best_state["val_acc"],
            },
            f,
            indent=2,
        )

    train_fused = gate * train_local.cpu() + (1.0 - gate) * train_global.cpu()
    val_fused = gate * val_local.cpu() + (1.0 - gate) * val_global.cpu()
    return train_fused, val_fused, gate


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    save_dir = args.save_dir / f"{args.data_set}_local_global_cbm_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    train_loader, train_labels = make_loader(args.anno_path, args.data_root, args)
    val_loader, val_labels = make_loader(args.val_anno_path, args.val_data_root or args.data_root, args)

    model_args = build_videomae_args(args)
    backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
    model = VideoMAELocalizer(backbone, out_channels=args.num_concepts).to(device)
    checkpoint = torch.load(args.localizer_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    train_features, train_y, train_sample_ids = extract_pooled_features(
        loader=train_loader,
        model=model,
        device=device,
        pooling=args.pooling,
        pool_source=args.pool_source,
        desc="extract-train",
    )
    val_features, val_y, val_sample_ids = extract_pooled_features(
        loader=val_loader,
        model=model,
        device=device,
        pooling=args.pooling,
        pool_source=args.pool_source,
        desc="extract-val",
    )

    if train_labels != train_y.tolist():
        raise ValueError("Train label order mismatch between dataset labels and extracted labels.")
    if val_labels != val_y.tolist():
        raise ValueError("Val label order mismatch between dataset labels and extracted labels.")

    torch.save(train_features, save_dir / "local_train_features.pt")
    torch.save(val_features, save_dir / "local_val_features.pt")

    train_global_labels, train_video_names = load_global_labels(args.global_label_dir / "hard_label_train.pkl")
    val_global_labels, val_video_names = load_global_labels(args.global_label_dir / "hard_label_val.pkl")
    validate_alignment(train_sample_ids, train_video_names, "train")
    validate_alignment(val_sample_ids, val_video_names, "val")
    if train_global_labels.shape[1] != args.num_concepts:
        raise ValueError(
            f"Global label dim {train_global_labels.shape[1]} does not match --num-concepts {args.num_concepts}"
        )

    concepts = [str(i) for i in range(train_global_labels.shape[1])]
    with open(save_dir / "concepts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(concepts))

    if args.fusion_mode in {"concat", "gated", "learnable_gated"}:
        if args.global_pose_dir is None or args.global_backbone_train is None or args.global_backbone_val is None:
            raise ValueError(
                f"--fusion-mode {args.fusion_mode} requires --global-pose-dir, --global-backbone-train, and --global-backbone-val."
            )

        train_local_norm, val_local_norm = normalize_concept_features(
            train_features=train_features,
            val_features=val_features,
            save_dir=save_dir,
            prefix="local",
        )
        train_global_feat = torch.load(args.global_backbone_train, map_location="cpu").float()
        val_global_feat = torch.load(args.global_backbone_val, map_location="cpu").float()
        if train_global_feat.shape[0] != train_features.shape[0]:
            raise ValueError(
                f"Train sample count mismatch: local={train_features.shape[0]}, global={train_global_feat.shape[0]}"
            )
        if val_global_feat.shape[0] != val_features.shape[0]:
            raise ValueError(
                f"Val sample count mismatch: local={val_features.shape[0]}, global={val_global_feat.shape[0]}"
            )
        train_global_norm, val_global_norm, global_concepts = build_global_pose_features(
            global_pose_dir=args.global_pose_dir,
            train_backbone_features=train_global_feat,
            val_backbone_features=val_global_feat,
            save_dir=save_dir,
        )
        if len(global_concepts) != train_local_norm.shape[1]:
            raise ValueError(
                f"Concept dim mismatch between local ({train_local_norm.shape[1]}) and global ({len(global_concepts)})"
            )

        torch.save(train_local_norm, save_dir / "local_train_features_norm.pt")
        torch.save(val_local_norm, save_dir / "local_val_features_norm.pt")
        torch.save(train_global_norm, save_dir / "global_train_features_norm.pt")
        torch.save(val_global_norm, save_dir / "global_val_features_norm.pt")

        if args.fusion_mode == "concat":
            fused_train = torch.cat([train_local_norm, train_global_norm], dim=1)
            fused_val = torch.cat([val_local_norm, val_global_norm], dim=1)
            fused_concepts = [f"local::{name}" for name in global_concepts] + [f"global::{name}" for name in global_concepts]
            with open(save_dir / "fused_concepts.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(fused_concepts))
        elif args.fusion_mode == "gated":
            if not 0.0 <= args.fusion_gate <= 1.0:
                raise ValueError(f"--fusion-gate must be in [0, 1], got {args.fusion_gate}")
            fused_train = args.fusion_gate * train_local_norm + (1.0 - args.fusion_gate) * train_global_norm
            fused_val = args.fusion_gate * val_local_norm + (1.0 - args.fusion_gate) * val_global_norm
            with open(save_dir / "fused_concepts.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(global_concepts))
            with open(save_dir / "fusion_config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fusion_mode": args.fusion_mode,
                        "fusion_gate": args.fusion_gate,
                        "local_weight": args.fusion_gate,
                        "global_weight": 1.0 - args.fusion_gate,
                    },
                    f,
                    indent=2,
                )
        else:
            fused_train, fused_val, gate = train_learnable_gate(
                args=args,
                train_local=train_local_norm,
                val_local=val_local_norm,
                train_global=train_global_norm,
                val_global=val_global_norm,
                train_y=train_y,
                val_y=val_y,
                save_dir=save_dir,
            )
            with open(save_dir / "fused_concepts.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(global_concepts))

        torch.save(fused_train, save_dir / "fused_train_features.pt")
        torch.save(fused_val, save_dir / "fused_val_features.pt")
        train_classifier_on_concepts(
            args=args,
            train_c=fused_train,
            val_c=fused_val,
            train_y=train_y,
            val_y=val_y,
            save_dir=save_dir,
        )
        return

    w_c, best_val_loss = train_global_concept_layer(
        args=args,
        train_features=train_features,
        val_features=val_features,
        train_labels=train_global_labels,
        val_labels=val_global_labels,
        save_dir=save_dir,
    )
    print(f"Concept layer best val loss: {best_val_loss:.6f}")

    train_c, val_c = train_classifier_identical(
        args=args,
        w_c=w_c,
        concepts=concepts,
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

"""Visualize contribution maps for Per-Concept Attention CBM.

For each test sample, produces side-by-side comparisons of:
  1. Original video frames
  2. Localizer heatmap (raw concept logits)
  3. Attention map (softmax of logits)
  4. Contribution map (attn × W_c relevance)

Only the top-K concepts (by W_g weight for the predicted class) are shown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from configs.defaults import build_videomae_args
from datasets.local_video_dataset import LocalVideoDataset
from models.attention_pool import ConceptGuidedAttentionPool
from models.backbone import FrozenVideoMAEBackbone
from models.localizer import VideoMAELocalizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize contribution maps for Per-Concept CBM.")
    # data
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # model
    parser.add_argument("--backbone", type=str, default="vmae_vit_base_patch16_224")
    parser.add_argument("--finetune", type=str, required=True)
    parser.add_argument("--data-set", type=str, required=True)
    parser.add_argument("--nb-classes", dest="nb_classes", type=int, required=True)
    parser.add_argument("--num-concepts", type=int, required=True)
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
    # checkpoints
    parser.add_argument("--localizer-ckpt", type=Path, required=True)
    parser.add_argument("--wc-path", type=Path, required=True, help="W_c.pt from Per-Concept CBM [C, D]")
    parser.add_argument("--wg-path", type=Path, required=True, help="W_g.pt [num_classes, C]")
    parser.add_argument("--bg-path", type=Path, default=None, help="b_g.pt [num_classes]")
    parser.add_argument("--proj-mean-path", type=Path, default=None)
    parser.add_argument("--proj-std-path", type=Path, default=None)
    # visualization
    parser.add_argument("--attention-temperature", type=float, default=5.0)
    parser.add_argument("--topk-concepts", type=int, default=5,
                        help="Show top-K concepts by W_g weight for predicted class")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--class-names", type=Path, default=None,
                        help="Path to class_list.txt")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay opacity")
    parser.set_defaults(use_checkpoint=False, use_mean_pooling=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_uint8_rgb(frame: torch.Tensor) -> np.ndarray:
    """Convert normalized [C, H, W] tensor to [H, W, 3] uint8."""
    frame = frame.cpu().float()
    frame = (frame * _STD + _MEAN).clamp(0, 1)
    return (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _normalize_map(spatial_map: torch.Tensor) -> torch.Tensor:
    """Normalize a spatial map to [0, 1] for visualization."""
    vmin, vmax = spatial_map.min(), spatial_map.max()
    if vmax - vmin < 1e-8:
        return torch.zeros_like(spatial_map)
    return (spatial_map - vmin) / (vmax - vmin)


def _upsample_map(spatial_map: torch.Tensor, size: int) -> torch.Tensor:
    """Upsample [T, H, W] map to [T, size, size]."""
    T = spatial_map.shape[0]
    return F.interpolate(
        spatial_map.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(1)  # [T, size, size]


def _apply_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply jet-like colormap to [H, W] float array in [0, 1]."""
    import matplotlib.cm as cm
    colored = cm.jet(gray)[:, :, :3]  # [H, W, 3] float 0-1
    return (colored * 255).astype(np.uint8)


def _apply_diverging_colormap(values: np.ndarray) -> np.ndarray:
    """Apply diverging colormap to [H, W] float array in [-1, 1].

    -1 = blue (suppresses concept), 0 = white/neutral, +1 = red (supports concept).
    """
    import matplotlib.cm as cm
    # Map [-1, 1] to [0, 1] for the colormap
    mapped = (values + 1.0) / 2.0
    mapped = np.clip(mapped, 0, 1)
    colored = cm.RdBu_r(mapped)[:, :, :3]  # Red=positive, Blue=negative
    return (colored * 255).astype(np.uint8)


def _overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """Blend frame [H,W,3] uint8 with heatmap [H,W,3] uint8."""
    blended = (1 - alpha) * frame.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    class_names = None
    if args.class_names is not None:
        with open(args.class_names) as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Load dataset
    dataset = LocalVideoDataset(
        anno_path=args.anno_path,
        data_root=args.data_root,
        data_set=args.data_set,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        input_size=args.input_size,
        deterministic=True,
    )

    # Load frozen localizer
    model_args = build_videomae_args(args)
    backbone = FrozenVideoMAEBackbone.from_args(model_args, device=device)
    model = VideoMAELocalizer(backbone, out_channels=args.num_concepts).to(device)
    ckpt = torch.load(args.localizer_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load W_c [C, D] and W_g [num_classes, C]
    w_c = torch.load(args.wc_path, map_location="cpu", weights_only=True).float()  # [C, D]
    w_g = torch.load(args.wg_path, map_location="cpu", weights_only=True).float()  # [num_classes, C]
    b_g = None
    if args.bg_path is not None:
        b_g = torch.load(args.bg_path, map_location="cpu", weights_only=True).float()

    # Load normalization stats if available
    proj_mean = proj_std = None
    if args.proj_mean_path is not None:
        proj_mean = torch.load(args.proj_mean_path, map_location="cpu", weights_only=True).float()
    if args.proj_std_path is not None:
        proj_std = torch.load(args.proj_std_path, map_location="cpu", weights_only=True).float().clamp_min(1e-6)

    pool = ConceptGuidedAttentionPool(temperature=args.attention_temperature)

    # Extract fc_norm
    vmae_model = model.backbone.model
    fc_norm = vmae_model.fc_norm if vmae_model.fc_norm is not None else torch.nn.Identity()
    fc_norm = fc_norm.to(device).eval()

    print(f"W_c: {tuple(w_c.shape)}, W_g: {tuple(w_g.shape)}")
    print(f"Temperature: {args.attention_temperature}")
    print(f"Visualizing up to {args.max_samples} samples, top-{args.topk_concepts} concepts")

    num_samples = min(len(dataset), args.max_samples)

    for idx in tqdm(range(num_samples), desc="visualize"):
        video, label, meta = dataset[idx]
        sample_id = meta["sample_id"]
        video_input = video.unsqueeze(0).to(device)  # [1, C, T_frames, H, W] -> need [1, C, T, H, W]

        with torch.no_grad():
            concept_logits, feature_map = model(video_input)
            # concept_logits: [1, C, T, H, W]
            # feature_map:    [1, D, T, H, W]

        B, C, T, H, W = concept_logits.shape
        D = feature_map.shape[1]
        S = T * H * W

        # --- Compute attention ---
        logit_flat = concept_logits.view(1, C, S)
        attn = torch.softmax(logit_flat / args.attention_temperature, dim=-1)  # [1, C, S]
        attn_spatial = attn.view(1, C, T, H, W)  # [1, C, T, H, W]

        # --- Per-concept pooled features ---
        pooled = pool(feature_map, concept_logits)  # [1, C, D]
        pooled_flat = pooled.view(C, D)
        pooled_flat = fc_norm(pooled_flat)  # LayerNorm per concept

        # --- Concept scores via per-concept W_c ---
        concept_scores = torch.einsum("cd, cd -> c", w_c, pooled_flat.cpu())  # [C]

        # --- Normalize concept scores if stats available ---
        if proj_mean is not None and proj_std is not None:
            concept_scores_norm = (concept_scores - proj_mean.squeeze()) / proj_std.squeeze()
        else:
            concept_scores_norm = concept_scores

        # --- Predict class ---
        class_logits = concept_scores_norm @ w_g.T  # [num_classes]
        if b_g is not None:
            class_logits = class_logits + b_g
        pred_class = class_logits.argmax().item()
        pred_name = class_names[pred_class] if class_names else str(pred_class)
        gt_name = class_names[label] if class_names else str(label)

        # --- Select top-K concepts by concept score magnitude ---
        topk_indices = concept_scores.abs().argsort(descending=True)[:args.topk_concepts].tolist()

        # --- Compute per-position relevance and contribution maps ---
        # relevance[c, t, h, w] = W_c[c] · backbone[t, h, w]
        feat_spatial = feature_map[0]  # [D, T, H, W]

        # Apply fc_norm per position
        feat_for_proj = feat_spatial.permute(1, 2, 3, 0).reshape(-1, D)  # [S, D]
        feat_for_proj = fc_norm(feat_for_proj)  # [S, D]

        # relevance[c, s] = W_c[c] · feat[s]
        relevance = torch.einsum("cd, sd -> cs", w_c.to(device), feat_for_proj)  # [C, S]
        relevance = relevance.view(C, T, H, W).cpu()  # [C, T, H, W]

        # contribution[c, t, h, w] = attn[c, t, h, w] × relevance[c, t, h, w]
        contribution = attn_spatial[0].cpu() * relevance  # [C, T, H, W]

        # --- Build visualization for each top concept ---
        sample_dir = output_dir / sample_id.replace("/", "__")
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(sample_dir / "info.json", "w") as f:
            json.dump({
                "sample_id": sample_id,
                "gt_label": label,
                "gt_name": gt_name,
                "pred_class": pred_class,
                "pred_name": pred_name,
                "concept_scores": concept_scores.tolist(),
                "topk_concepts": topk_indices,
                "wg_weights": {str(c): float(w_g[pred_class, c]) for c in topk_indices},
            }, f, indent=2)

        # Original frames (from input video tensor, tubelet_size=2 so T_grid = T_frames/2)
        # video: [C, T_frames, H_input, W_input]
        input_size = args.input_size

        for concept_idx in topk_indices:
            wg_weight = w_g[pred_class, concept_idx].item()
            c_score = concept_scores[concept_idx].item()

            # Get spatial maps for this concept
            logit_map = concept_logits[0, concept_idx].cpu()       # [T, H_grid, W_grid]
            attn_map = attn_spatial[0, concept_idx].cpu()           # [T, H_grid, W_grid]
            contrib_map = contribution[concept_idx]                  # [T, H_grid, W_grid]

            # Absolute scale: no per-map normalization
            # Localizer: sigmoid output, already in [0, 1]
            logit_vis = torch.sigmoid(logit_map).clamp(0, 1)
            # Attention: softmax output, scale relative to uniform baseline
            # uniform = 1/S; values >> 1/S are "high attention"
            # Multiply by S so uniform = 1.0, then clamp to [0, 3] and divide by 3
            uniform_val = 1.0 / S
            attn_vis = (attn_map / uniform_val).clamp(0, 3) / 3.0
            # Contribution: keep sign, scale by 95th percentile of abs values
            # Positive = supports concept, Negative = suppresses concept
            p95 = torch.quantile(contribution.abs().float(), 0.95) if contribution.abs().max() > 0 else 1.0
            contrib_vis = (contrib_map / max(float(p95), 1e-8)).clamp(-1, 1)  # [-1, 1]

            # Upsample to input resolution
            logit_up = _upsample_map(logit_vis, input_size)    # [T, 224, 224]
            attn_up = _upsample_map(attn_vis, input_size)      # [T, 224, 224]
            contrib_up = _upsample_map(contrib_vis, input_size) # [T, 224, 224]

            # Build image grid: each row is one timestep
            # Columns: original | localizer | attention | contribution
            rows = []
            for t in range(T):
                # Map grid time t back to video frame index
                # tubelet_size=2, so grid time t corresponds to frame t*2
                frame_idx = min(t * args.tubelet_size, video.shape[1] - 1)
                frame = _to_uint8_rgb(video[:, frame_idx])  # [H, W, 3]

                logit_hm = _apply_colormap(logit_up[t].numpy())
                attn_hm = _apply_colormap(attn_up[t].numpy())
                contrib_hm = _apply_diverging_colormap(contrib_up[t].numpy())

                col_orig = frame
                col_logit = _overlay_heatmap(frame, logit_hm, args.alpha)
                col_attn = _overlay_heatmap(frame, attn_hm, args.alpha)
                col_contrib = _overlay_heatmap(frame, contrib_hm, args.alpha)

                row = np.concatenate([col_orig, col_logit, col_attn, col_contrib], axis=1)
                rows.append(row)

            grid = np.concatenate(rows, axis=0)
            img = Image.fromarray(grid)

            fname = f"concept{concept_idx:02d}_wg{wg_weight:+.3f}_score{c_score:.3f}.png"
            img.save(sample_dir / fname)

        # Save a legend image
        legend_text = (
            f"Sample: {sample_id}\n"
            f"GT: {gt_name} | Pred: {pred_name}\n"
            f"Columns: Original | Localizer (sigmoid) | Attention (softmax) | Contribution\n"
            f"Top concepts for '{pred_name}':\n"
        )
        for c in topk_indices:
            legend_text += f"  concept {c}: W_g={w_g[pred_class, c]:.3f}, score={concept_scores[c]:.3f}\n"
        with open(sample_dir / "legend.txt", "w") as f:
            f.write(legend_text)

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()

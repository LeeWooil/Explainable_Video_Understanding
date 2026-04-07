"""Visualize contribution maps for Per-Concept Attention CBM (paper figure).

For each test sample, produces a single unified figure with:
  (a) Horizontal bar chart of per-concept contributions to the predicted class
  (b) Spatial contribution maps for top-K concepts at peak timesteps

Only the top-K concepts (by W_g weight for the predicted class) are shown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
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
    parser.add_argument("--peak-timesteps", type=int, default=2,
                        help="Number of peak timesteps to show per concept")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--class-names", type=Path, default=None,
                        help="Path to class_list.txt")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay opacity")
    parser.add_argument("--concept-canvas-dir", type=Path, default=None,
                        help="Directory with concept_NNN_canvas.png trajectory visualizations. "
                             "If provided, trajectory patterns are overlaid on contribution maps.")
    parser.add_argument("--canvas-alpha", type=float, default=0.6,
                        help="Opacity for trajectory canvas overlay on contribution maps")
    parser.add_argument("--trajectory-data", type=Path, default=None,
                        help="Path to trajectories.npy [num_concepts, L, 2] displacement vectors. "
                             "If provided, arrow-shaped trajectories are rendered on contribution maps.")
    parser.add_argument("--trajectory-scale", type=float, default=3.0,
                        help="Scale factor for trajectory displacements when rendering arrows.")
    parser.add_argument("--arrow-line-width", type=int, default=4,
                        help="Line width for arrow-shaped trajectory rendering.")
    parser.add_argument("--arrow-size", type=float, default=12.0,
                        help="Arrowhead size for trajectory rendering.")
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


def _load_concept_canvas(canvas_dir: Path, concept_idx: int, size: int) -> np.ndarray | None:
    """Load concept_NNN_canvas.png, resize to (size, size), return [H,W,3] uint8 or None."""
    path = canvas_dir / f"concept_{concept_idx:03d}_canvas.png"
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


def _overlay_canvas(
    base: np.ndarray,
    canvas: np.ndarray,
    alpha: float,
    contrib_map: np.ndarray,
) -> np.ndarray:
    """Overlay trajectory canvas centered on the contribution map's peak activation.

    1. Find the centroid of the canvas trajectory (non-black pixels).
    2. Find the centroid of the peak activation region in the contribution map.
    3. Translate the canvas so the trajectory centroid aligns with the activation centroid.
    4. Blend only non-black canvas pixels onto the base image.
    """
    H, W = base.shape[:2]
    # Canvas trajectory mask
    canvas_mask = canvas.max(axis=2) > 15  # [H, W] bool

    if not canvas_mask.any():
        return base

    # Centroid of canvas trajectory
    cy_canvas, cx_canvas = np.argwhere(canvas_mask).mean(axis=0)

    # Centroid of peak activation (only positive contributions — where the concept is detected)
    pos_contrib = np.maximum(contrib_map, 0)  # [H, W]
    total = pos_contrib.sum()
    if total < 1e-8:
        # No meaningful activation — fall back to image center
        cy_act, cx_act = H / 2.0, W / 2.0
    else:
        ys, xs = np.mgrid[:H, :W]
        cy_act = (ys * pos_contrib).sum() / total
        cx_act = (xs * pos_contrib).sum() / total

    # Compute translation offset
    dy = int(round(cy_act - cy_canvas))
    dx = int(round(cx_act - cx_canvas))

    # Translate canvas and mask
    shifted_canvas = np.zeros_like(canvas)
    shifted_mask = np.zeros((H, W), dtype=np.float32)

    # Source and destination slicing
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)

    h_copy = min(src_y1 - src_y0, dst_y1 - dst_y0)
    w_copy = min(src_x1 - src_x0, dst_x1 - dst_x0)
    if h_copy <= 0 or w_copy <= 0:
        return base

    shifted_canvas[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy] = \
        canvas[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy]
    shifted_mask[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy] = \
        canvas_mask[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy].astype(np.float32)

    mask3 = shifted_mask[:, :, None]  # [H, W, 1]
    blended = base.astype(np.float32) * (1 - mask3 * alpha) + shifted_canvas.astype(np.float32) * mask3 * alpha
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Arrow-shaped trajectory rendering (ported from trajectory labeling script)
# ---------------------------------------------------------------------------

def _build_cumulative_points(trajectory: np.ndarray) -> np.ndarray:
    """Convert displacement vectors [L, 2] to cumulative positions [L+1, 2]."""
    return np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), np.cumsum(trajectory, axis=0)],
        axis=0,
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    line_width: int = 4,
    arrow_size: float = 12.0,
) -> None:
    """Draw a line segment with an arrowhead from start to end."""
    draw.line(
        [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))],
        fill=color,
        width=line_width,
    )

    vec = end - start
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return

    direction = vec / norm
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    head_len = min(arrow_size, max(norm * 0.6, arrow_size * 0.5))
    base = end - direction * head_len
    left = base + perp * (head_len * 0.45)
    right = base - perp * (head_len * 0.45)
    draw.polygon(
        [
            (float(end[0]), float(end[1])),
            (float(left[0]), float(left[1])),
            (float(right[0]), float(right[1])),
        ],
        fill=color,
    )


def _overlay_trajectory_arrows(
    base: np.ndarray,
    trajectory: np.ndarray,
    anchor_xy: tuple[float, float],
    scale: float = 3.0,
    line_width: int = 4,
    arrow_size: float = 12.0,
) -> np.ndarray:
    """Render arrow-shaped trajectory on the base image, anchored at a given point.

    Parameters
    ----------
    base : [H, W, 3] uint8 image.
    trajectory : [L, 2] displacement vectors for this concept.
    anchor_xy : (x, y) pixel coordinates to center the trajectory on.
    scale : Scale factor for trajectory displacements.
    line_width : Arrow shaft width.
    arrow_size : Arrowhead size.

    Returns
    -------
    [H, W, 3] uint8 blended image.
    """
    anchor = np.array(anchor_xy, dtype=np.float32)

    # Build cumulative trajectory points, scaled and anchored
    points = _build_cumulative_points(trajectory) * float(scale)
    # Center the trajectory on the anchor
    traj_center = points.mean(axis=0)
    points = points - traj_center + anchor[None, :]

    # Draw arrows on a PIL image
    img = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(img)

    total_steps = len(trajectory)
    for idx in range(1, total_steps + 1):
        start = points[idx - 1]
        end = points[idx]
        # Color gradient: green-ish at start → red-ish at end
        ratio = idx / max(total_steps, 1)
        color = (
            int(25 + 200 * ratio),
            int(220 - 90 * ratio),
            int(70 + 140 * (1.0 - ratio)),
        )
        _draw_arrow(draw, start, end, color=color, line_width=line_width, arrow_size=arrow_size)

    # Draw start point marker
    r = 5
    draw.ellipse(
        [
            float(points[0][0] - r),
            float(points[0][1] - r),
            float(points[0][0] + r),
            float(points[0][1] + r),
        ],
        fill=(255, 235, 59),
        outline=(0, 0, 0),
        width=1,
    )

    return np.asarray(img, dtype=np.uint8)


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

    # Pre-load concept canvas images if available
    concept_canvases: dict[int, np.ndarray | None] = {}
    if args.concept_canvas_dir is not None:
        print(f"Loading concept canvases from {args.concept_canvas_dir}")

    # Load trajectory displacement data if available
    trajectory_data: np.ndarray | None = None
    if args.trajectory_data is not None:
        trajectory_data = np.load(args.trajectory_data).astype(np.float32)
        print(f"Loaded trajectory data: {trajectory_data.shape} from {args.trajectory_data}")

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

        # --- Select top-K concepts by class-level contribution to the predicted class ---
        class_contrib = concept_scores * w_g[pred_class]
        topk_indices = class_contrib.argsort(descending=True)[:args.topk_concepts].tolist()

        # --- Compute per-position relevance and contribution maps ---
        # relevance[c, t, h, w] = W_c[c] · backbone[t, h, w]
        feat_spatial = feature_map[0]  # [D, T, H, W]

        feat_for_proj = feat_spatial.permute(1, 2, 3, 0).reshape(-1, D)  # [S, D]

        # relevance[c, s] = W_c[c] · feat[s]
        relevance = torch.einsum("cd, sd -> cs", w_c.to(device), feat_for_proj)  # [C, S]
        relevance = relevance.view(C, T, H, W).cpu()  # [C, T, H, W]

        # contribution[c, t, h, w] = attn[c, t, h, w] × relevance[c, t, h, w]
        contribution = attn_spatial[0].cpu() * relevance  # [C, T, H, W]

        # --- Build unified paper figure ---
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
                "class_contributions": class_contrib.tolist(),
                "topk_concepts": topk_indices,
                "wg_weights": {str(c): float(w_g[pred_class, c]) for c in topk_indices},
            }, f, indent=2)

        input_size = args.input_size
        K = len(topk_indices)
        n_peak = args.peak_timesteps

        # Global contribution scale (95th percentile across all concepts)
        p95 = torch.quantile(contribution.abs().float(), 0.95) if contribution.abs().max() > 0 else 1.0
        p95 = max(float(p95), 1e-8)

        # Pre-compute per-concept data: peak (t, h, w) positions + upsampled maps
        concept_vis_data = []
        for concept_idx in topk_indices:
            contrib_map = contribution[concept_idx]  # [T, H_grid, W_grid]

            # Find top-N peak positions jointly in (T, H, W) by absolute contribution
            flat_abs = contrib_map.abs().reshape(-1)
            topk_flat = flat_abs.argsort(descending=True)
            # Extract unique timesteps from top positions
            seen_ts: set[int] = set()
            peak_positions: list[tuple[int, int, int]] = []  # (t, h_grid, w_grid)
            for fi in topk_flat.tolist():
                t_i = fi // (H * W)
                hw = fi % (H * W)
                h_i = hw // W
                w_i = hw % W
                if t_i not in seen_ts:
                    seen_ts.add(t_i)
                    peak_positions.append((t_i, h_i, w_i))
                    if len(peak_positions) >= n_peak:
                        break
            # Sort by timestep for chronological display
            peak_positions.sort(key=lambda x: x[0])

            contrib_vis = (contrib_map / p95).clamp(-1, 1)
            contrib_up = _upsample_map(contrib_vis, input_size)  # [T, 224, 224]

            concept_vis_data.append({
                "concept_idx": concept_idx,
                "peak_positions": peak_positions,
                "contrib_up": contrib_up,
                "wg_weight": w_g[pred_class, concept_idx].item(),
                "c_score": concept_scores[concept_idx].item(),
                "c_contrib": class_contrib[concept_idx].item(),
            })

        # --- Build figure with gridspec ---
        # Layout: top row = bar chart spanning full width
        #         bottom rows = K concepts × (n_peak × 2 columns: orig + contrib)
        fig = plt.figure(figsize=(3.0 * n_peak * 2, 2.5 + 2.5 * K), dpi=150)
        gs = gridspec.GridSpec(
            K + 1, n_peak * 2,
            figure=fig,
            height_ratios=[1.2] + [1.0] * K,
            hspace=0.35, wspace=0.08,
        )

        # (a) Bar chart of concept contributions
        ax_bar = fig.add_subplot(gs[0, :])
        contribs = [d["c_contrib"] for d in concept_vis_data]
        labels = [f"C{d['concept_idx']}" for d in concept_vis_data]
        colors = ["#d73027" if v >= 0 else "#4575b4" for v in contribs]
        y_pos = np.arange(K)
        ax_bar.barh(y_pos, contribs, color=colors, edgecolor="k", linewidth=0.5)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(labels, fontsize=10)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Contribution to predicted class", fontsize=10)
        ax_bar.axvline(0, color="k", linewidth=0.5)
        ax_bar.set_title(
            f"GT: {gt_name}  |  Pred: {pred_name}",
            fontsize=12, fontweight="bold",
        )
        ax_bar.tick_params(labelsize=9)

        # (b) Spatial maps: each row = one concept, columns = peak timesteps × (orig, contrib)
        import matplotlib.cm as cm

        for row_i, d in enumerate(concept_vis_data):
            peak_positions = d["peak_positions"]
            contrib_up = d["contrib_up"]
            concept_idx = d["concept_idx"]

            # Load canvas for this concept (lazy, cached)
            if args.concept_canvas_dir is not None and concept_idx not in concept_canvases:
                concept_canvases[concept_idx] = _load_concept_canvas(
                    args.concept_canvas_dir, concept_idx, input_size,
                )
            canvas = concept_canvases.get(concept_idx)

            for col_j, (t, h_grid, w_grid) in enumerate(peak_positions):
                # Convert grid-level peak (h_grid, w_grid) to pixel coordinates
                anchor_x = (w_grid + 0.5) * input_size / W
                anchor_y = (h_grid + 0.5) * input_size / H

                frame_idx = min(t * args.tubelet_size, video.shape[1] - 1)
                frame = _to_uint8_rgb(video[:, frame_idx])  # [H, W, 3]
                contrib_hm = _apply_diverging_colormap(contrib_up[t].numpy())
                overlay = _overlay_heatmap(frame, contrib_hm, args.alpha)
                # Overlay arrow-shaped trajectory if trajectory data is available
                if trajectory_data is not None and concept_idx < trajectory_data.shape[0]:
                    overlay = _overlay_trajectory_arrows(
                        overlay,
                        trajectory_data[concept_idx],
                        (anchor_x, anchor_y),
                        scale=args.trajectory_scale,
                        line_width=args.arrow_line_width,
                        arrow_size=args.arrow_size,
                    )
                # Fall back to canvas overlay if no trajectory data
                elif canvas is not None:
                    overlay = _overlay_canvas(
                        overlay, canvas, args.canvas_alpha,
                        contrib_up[t].numpy(),
                    )

                # Original frame
                ax_orig = fig.add_subplot(gs[row_i + 1, col_j * 2])
                ax_orig.imshow(frame)
                ax_orig.set_xticks([])
                ax_orig.set_yticks([])
                if col_j == 0:
                    ax_orig.set_ylabel(
                        f"C{concept_idx}\n({d['c_contrib']:+.2f})",
                        fontsize=9, rotation=0, labelpad=40, va="center",
                    )
                if row_i == 0:
                    ax_orig.set_title(f"t={t} orig", fontsize=8)

                # Contribution overlay
                ax_cont = fig.add_subplot(gs[row_i + 1, col_j * 2 + 1])
                ax_cont.imshow(overlay)
                ax_cont.set_xticks([])
                ax_cont.set_yticks([])
                if row_i == 0:
                    ax_cont.set_title(f"t={t} contrib", fontsize=8)

        fig.savefig(sample_dir / "paper_figure.png", bbox_inches="tight", pad_inches=0.1)
        fig.savefig(sample_dir / "paper_figure.pdf", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()

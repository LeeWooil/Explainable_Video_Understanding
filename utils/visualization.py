from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
_PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
    ],
    dtype=np.uint8,
)


def _to_uint8_rgb(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().to(torch.float32)
    frame = (frame * _STD) + _MEAN
    frame = frame.clamp(0.0, 1.0)
    frame = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return frame


def _mask_to_heatmap(mask: torch.Tensor) -> np.ndarray:
    arr = mask.detach().cpu().to(torch.float32).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    heatmap = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = (arr * 255.0).astype(np.uint8)
    heatmap[..., 1] = (arr * 64.0).astype(np.uint8)
    return heatmap


def _resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(size, size),
        mode="nearest",
    )[0, 0]


def _overlay(frame: np.ndarray, mask: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    heatmap = _mask_to_heatmap(mask)
    blended = ((1.0 - alpha) * frame.astype(np.float32) + alpha * heatmap.astype(np.float32)).clip(0, 255)
    return blended.astype(np.uint8)


def _select_concept_indices(
    tensor: torch.Tensor,
    topk: int,
) -> list[int]:
    flat = tensor.reshape(tensor.shape[0], -1).sum(dim=1)
    active = torch.where(flat > 0)[0]
    if len(active) == 0:
        return [int(flat.argmax().item())]
    if len(active) <= topk:
        return [int(idx.item()) for idx in active]
    top_vals, top_idx = torch.topk(flat, k=topk)
    return [int(idx.item()) for idx in top_idx]


def _multi_mask_overlay(
    frame: np.ndarray,
    masks: list[torch.Tensor],
    colors: list[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    if not masks:
        return frame
    mask_stack = torch.stack([mask.to(torch.float32) for mask in masks], dim=0)
    max_vals, assignments = torch.max(mask_stack, dim=0)
    overlay = frame.astype(np.float32).copy()
    for idx, color in enumerate(colors):
        active = (assignments == idx) & (max_vals > 0)
        if not torch.any(active):
            continue
        active_np = active.cpu().numpy()
        overlay[active_np] = (1.0 - alpha) * overlay[active_np] + alpha * color.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _winner_take_all_overlay(
    frame: np.ndarray,
    all_masks: torch.Tensor,
    selected_indices: list[int],
    colors: list[np.ndarray],
    alpha: float = 0.45,
    min_value: float = 0.0,
    scale_alpha_by_value: bool = False,
) -> np.ndarray:
    if len(selected_indices) == 0:
        return frame
    values, winners = torch.max(all_masks.to(torch.float32), dim=0)
    overlay = frame.astype(np.float32).copy()
    for local_idx, concept_idx in enumerate(selected_indices):
        active = (winners == concept_idx) & (values > min_value)
        if not torch.any(active):
            continue
        active_np = active.cpu().numpy()
        if scale_alpha_by_value:
            weights = (alpha * values[active].clamp(0.0, 1.0)).cpu().numpy().astype(np.float32)
            base_pixels = overlay[active_np]
            color = colors[local_idx].astype(np.float32)[None, :]
            overlay[active_np] = (1.0 - weights[:, None]) * base_pixels + weights[:, None] * color
        else:
            overlay[active_np] = (1.0 - alpha) * overlay[active_np] + alpha * colors[local_idx].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _top_patch_per_concept_overlay(
    frame: np.ndarray,
    all_masks: torch.Tensor,
    selected_indices: list[int],
    colors: list[np.ndarray],
    alpha: float = 0.65,
) -> np.ndarray:
    overlay = frame.astype(np.float32).copy()
    masks = all_masks.to(torch.float32)
    _, height, width = masks.shape

    for local_idx, concept_idx in enumerate(selected_indices):
        concept_mask = masks[concept_idx]
        spatial_idx = int(concept_mask.reshape(-1).argmax().item())
        row = spatial_idx // width
        col = spatial_idx % width
        color = colors[local_idx].astype(np.float32)

        row_start = int(np.floor((row / height) * overlay.shape[0]))
        row_end = int(np.ceil(((row + 1) / height) * overlay.shape[0]))
        col_start = int(np.floor((col / width) * overlay.shape[1]))
        col_end = int(np.ceil(((col + 1) / width) * overlay.shape[1]))
        overlay[row_start:row_end, col_start:col_end] = (
            (1.0 - alpha) * overlay[row_start:row_end, col_start:col_end] + alpha * color
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_localization_previews(
    videos: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    metas: Sequence[dict],
    output_dir: str | Path,
    epoch: int,
    max_samples: int = 4,
    threshold: float = 0.5,
) -> None:
    out_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(logits.detach())
    preds = (probs >= threshold).to(dtype=torch.float32)
    max_items = min(len(metas), max_samples)

    for sample_idx in range(max_items):
        video = videos[sample_idx].detach().cpu()
        concept_idx = int(targets[sample_idx].reshape(targets[sample_idx].shape[0], -1).sum(dim=1).argmax().item())
        target = targets[sample_idx, concept_idx].detach().cpu()
        prob = probs[sample_idx, concept_idx].detach().cpu()
        pred = preds[sample_idx, concept_idx].detach().cpu()
        sample_id = metas[sample_idx]["sample_id"].replace("/", "__")

        num_steps = min(video.shape[1], target.shape[0])
        rows = []
        for time_idx in range(num_steps):
            frame = _to_uint8_rgb(video[:, min(time_idx * 2, video.shape[1] - 1)])
            target_up = _resize_mask(target[time_idx], frame.shape[0])
            prob_up = _resize_mask(prob[time_idx], frame.shape[0])
            pred_up = _resize_mask(pred[time_idx], frame.shape[0])

            tiles = [
                frame,
                _overlay(frame, target_up),
                _overlay(frame, prob_up),
                _overlay(frame, pred_up),
            ]
            rows.append(np.concatenate(tiles, axis=1))

        if not rows:
            continue

        grid = np.concatenate(rows, axis=0)
        Image.fromarray(grid).save(out_dir / f"{sample_id}_concept_{concept_idx:03d}.png")


def save_multiconcept_localization_previews(
    videos: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    metas: Sequence[dict],
    output_dir: str | Path,
    tag: str,
    max_samples: int = 4,
    threshold: float = 0.5,
    topk_concepts: int = 3,
    concept_source: str = "target",
    concept_names: Sequence[str] | None = None,
    save_gt_frames: bool = False,
) -> None:
    out_dir = Path(output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(logits.detach())
    preds = (probs >= threshold).to(dtype=torch.float32)
    max_items = min(len(metas), max_samples)

    for sample_idx in range(max_items):
        video = videos[sample_idx].detach().cpu()
        sample_id = metas[sample_idx]["sample_id"].replace("/", "__")

        selector = targets[sample_idx] if concept_source == "target" else probs[sample_idx]
        concept_indices = _select_concept_indices(selector.detach().cpu(), topk=topk_concepts)
        colors = [_PALETTE[idx % len(_PALETTE)] for idx in range(len(concept_indices))]

        num_steps = min(video.shape[1], targets.shape[2])
        rows = []
        for time_idx in range(num_steps):
            frame = _to_uint8_rgb(video[:, min(time_idx * 2, video.shape[1] - 1)])
            coarse_prob_masks = probs[sample_idx, :, time_idx].detach().cpu()
            all_target_masks = torch.stack(
                [_resize_mask(targets[sample_idx, concept_idx, time_idx].detach().cpu(), frame.shape[0]) for concept_idx in range(targets.shape[1])],
                dim=0,
            )
            all_prob_masks = torch.stack(
                [_resize_mask(probs[sample_idx, concept_idx, time_idx].detach().cpu(), frame.shape[0]) for concept_idx in range(probs.shape[1])],
                dim=0,
            )
            all_pred_masks = torch.stack(
                [_resize_mask(preds[sample_idx, concept_idx, time_idx].detach().cpu(), frame.shape[0]) for concept_idx in range(preds.shape[1])],
                dim=0,
            )

            tiles = [
                frame,
                _winner_take_all_overlay(frame, all_target_masks, concept_indices, colors, min_value=0.0),
                _winner_take_all_overlay(
                    frame,
                    all_prob_masks,
                    concept_indices,
                    colors,
                    min_value=0.0,
                    scale_alpha_by_value=True,
                ),
                _winner_take_all_overlay(frame, all_pred_masks, concept_indices, colors, min_value=0.0),
                _top_patch_per_concept_overlay(frame, coarse_prob_masks, concept_indices, colors),
            ]
            rows.append(np.concatenate(tiles, axis=1))

            if save_gt_frames:
                gt_frame_dir = out_dir / f"{sample_id}_gt_frames"
                gt_frame_dir.mkdir(parents=True, exist_ok=True)
                gt_overlay = _winner_take_all_overlay(frame, all_target_masks, concept_indices, colors, min_value=0.0)
                Image.fromarray(gt_overlay).save(gt_frame_dir / f"frame_{time_idx:03d}.png")

        if not rows:
            continue

        grid = np.concatenate(rows, axis=0)
        Image.fromarray(grid).save(out_dir / f"{sample_id}_multiconcept.png")

        legend = []
        for local_idx, concept_idx in enumerate(concept_indices):
            concept_name = str(concept_idx) if concept_names is None or concept_idx >= len(concept_names) else concept_names[concept_idx]
            legend.append(
                {
                    "concept_idx": concept_idx,
                    "concept_name": concept_name,
                    "color_rgb": colors[local_idx].tolist(),
                }
            )
        with open(out_dir / f"{sample_id}_multiconcept.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_id": metas[sample_idx]["sample_id"],
                    "concept_source": concept_source,
                    "topk_concepts": topk_concepts,
                    "legend": legend,
                },
                f,
                indent=2,
            )

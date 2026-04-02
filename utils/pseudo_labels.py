from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def load_pixel_mask(mask_root: str | Path, sample_id: str) -> np.ndarray:
    mask_path = Path(mask_root) / sample_id / "pixel_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Pixel mask not found: {mask_path}")
    mask = np.load(mask_path, mmap_mode="r")
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Expected pixel mask with shape [T,H,W] or [C,T,H,W], got {tuple(mask.shape)}")


def load_metadata(mask_root: str | Path, sample_id: str) -> dict:
    metadata_path = Path(mask_root) / sample_id / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_mask_by_frame_indices(mask: np.ndarray, frame_indices: Sequence[int]) -> np.ndarray:
    if len(frame_indices) == 0:
        raise ValueError("frame_indices must be non-empty.")
    if mask.ndim == 3:
        clamped_indices = [max(0, min(int(idx), mask.shape[0] - 1)) for idx in frame_indices]
        return mask[clamped_indices]
    if mask.ndim == 4:
        clamped_indices = [max(0, min(int(idx), mask.shape[1] - 1)) for idx in frame_indices]
        return mask[:, clamped_indices]
    raise ValueError(f"Expected [T,H,W] or [C,T,H,W], got {tuple(mask.shape)}")


def apply_spatial_transform_to_mask(
    mask: torch.Tensor,
    crop_params: tuple[int, int, int, int],
    input_size: int,
) -> torch.Tensor:
    if mask.ndim not in (3, 4):
        raise ValueError(f"Expected [T,H,W] or [C,T,H,W], got {tuple(mask.shape)}")
    top, left, crop_h, crop_w = crop_params
    if mask.ndim == 3:
        cropped = mask[:, top : top + crop_h, left : left + crop_w]
        resized = F.interpolate(cropped.unsqueeze(1), size=(input_size, input_size), mode="nearest")
        return resized.squeeze(1)

    num_concepts, num_frames = mask.shape[:2]
    cropped = mask[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(
        cropped.reshape(num_concepts * num_frames, 1, crop_h, crop_w),
        size=(input_size, input_size),
        mode="nearest",
    )
    return resized.reshape(num_concepts, num_frames, input_size, input_size)


def spatial_pool_to_patches(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    if mask.ndim == 3:
        pooled = F.max_pool2d(mask.unsqueeze(1), kernel_size=patch_size, stride=patch_size)
        return pooled.squeeze(1)
    if mask.ndim == 4:
        num_concepts, num_frames, height, width = mask.shape
        pooled = F.max_pool2d(
            mask.reshape(num_concepts * num_frames, 1, height, width),
            kernel_size=patch_size,
            stride=patch_size,
        )
        return pooled.reshape(num_concepts, num_frames, pooled.shape[-2], pooled.shape[-1])
    raise ValueError(f"Expected [T,H,W] or [C,T,H,W], got {tuple(mask.shape)}")


def load_selected_pixel_mask(
    mask_root: str | Path,
    sample_id: str,
    frame_indices: Sequence[int],
    crop_params: tuple[int, int, int, int],
) -> torch.Tensor:
    mask = load_pixel_mask(mask_root, sample_id)
    selected = select_mask_by_frame_indices(mask, frame_indices)
    top, left, crop_h, crop_w = crop_params

    if selected.ndim == 3:
        cropped = selected[:, top : top + crop_h, left : left + crop_w]
    elif selected.ndim == 4:
        cropped = selected[:, :, top : top + crop_h, left : left + crop_w]
    else:
        raise ValueError(f"Expected [T,H,W] or [C,T,H,W], got {tuple(selected.shape)}")

    # Materialize only the sliced region so we avoid loading the full .npy into RAM.
    return torch.from_numpy(np.array(cropped, copy=True)).float()


def pool_mask_to_tubelets(mask: torch.Tensor, tubelet_size: int) -> torch.Tensor:
    if mask.ndim == 3:
        time_steps = mask.shape[0]
        usable = (time_steps // tubelet_size) * tubelet_size
        if usable <= 0:
            raise ValueError(f"time_steps={time_steps} too small for tubelet_size={tubelet_size}")
        mask = mask[:usable]
        return mask.view(usable // tubelet_size, tubelet_size, mask.shape[1], mask.shape[2]).amax(dim=1)
    if mask.ndim == 4:
        num_concepts, time_steps, height, width = mask.shape
        usable = (time_steps // tubelet_size) * tubelet_size
        if usable <= 0:
            raise ValueError(f"time_steps={time_steps} too small for tubelet_size={tubelet_size}")
        mask = mask[:, :usable]
        return mask.view(num_concepts, usable // tubelet_size, tubelet_size, height, width).amax(dim=2)
    raise ValueError(f"Expected [T,H,W] or [C,T,H,W], got {tuple(mask.shape)}")


def build_batch_targets(
    metas: Sequence[dict],
    mask_root: str | Path,
    tubelet_size: int,
    input_size: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    targets = []
    for meta in metas:
        targets.append(
            build_target_from_meta(
                meta=meta,
                mask_root=mask_root,
                tubelet_size=tubelet_size,
                input_size=input_size,
                patch_size=patch_size,
            )
        )
    target_tensor = torch.stack(targets, dim=0)  # [B,C,T,H,W]
    return target_tensor.to(device=device, dtype=torch.float32)


def build_target_from_meta(
    meta: dict,
    mask_root: str | Path,
    tubelet_size: int,
    input_size: int,
    patch_size: int,
) -> torch.Tensor:
    sample_id = meta["sample_id"]
    frame_indices = meta["frame_indices"]
    crop_params = tuple(meta["crop_params"])
    selected = load_selected_pixel_mask(
        mask_root=mask_root,
        sample_id=sample_id,
        frame_indices=frame_indices,
        crop_params=crop_params,
    )
    resized = apply_spatial_transform_to_mask(
        selected,
        crop_params=(0, 0, selected.shape[-2], selected.shape[-1]),
        input_size=input_size,
    )
    patch_mask = spatial_pool_to_patches(resized, patch_size=patch_size)
    pooled = pool_mask_to_tubelets(patch_mask, tubelet_size=tubelet_size)
    if pooled.ndim == 3:
        pooled = pooled.unsqueeze(0)
    return pooled.to(dtype=torch.float32)

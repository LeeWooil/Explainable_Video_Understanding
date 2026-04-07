from __future__ import annotations

import os
import random
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import video_transforms as videomae_video_transforms
from utils import volume_transforms
from utils.pseudo_labels import build_target_from_meta


def _resolve_sample_path(data_root: str | Path, sample: str) -> str:
    sample_path = Path(sample)
    if sample_path.is_absolute():
        return str(sample_path)
    return str(Path(data_root) / sample)


def _build_sample_id(sample: str) -> str:
    sample_path = Path(sample)
    no_suffix = sample_path.with_suffix("")
    return no_suffix.as_posix()


def _tensor_normalize(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _apply_crop_and_resize(images: torch.Tensor, crop_params: tuple[int, int, int, int], target_size: int) -> torch.Tensor:
    top, left, crop_h, crop_w = crop_params
    cropped = images[:, :, top : top + crop_h, left : left + crop_w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


def _compute_resize_and_center_crop_params(
    height: int,
    width: int,
    short_side_size: int,
    input_size: int,
) -> tuple[int, int, int, int]:
    if (width <= height and width == short_side_size) or (height <= width and height == short_side_size):
        new_h, new_w = height, width
    elif width < height:
        new_w = short_side_size
        new_h = int(short_side_size * height / width)
    else:
        new_h = short_side_size
        new_w = int(short_side_size * width / height)

    crop_h = crop_w = input_size
    top = int(round((new_h - crop_h) / 2.0))
    left = int(round((new_w - crop_w) / 2.0))
    return top, left, crop_h, crop_w


class LocalVideoDataset(Dataset):
    def __init__(
        self,
        anno_path: str | Path,
        data_root: str | Path,
        data_set: str,
        num_frames: int = 16,
        sampling_rate: int = 4,
        input_size: int = 224,
        short_side_size: int = 224,
        deterministic: bool = True,
        view_mode: str = "random",
    ) -> None:
        self.anno_path = str(anno_path)
        self.data_root = str(data_root)
        self.data_set = data_set
        self.num_frames = int(num_frames)
        self.sampling_rate = int(sampling_rate)
        self.input_size = int(input_size)
        self.short_side_size = int(short_side_size)
        self.deterministic = bool(deterministic)
        self.view_mode = str(view_mode)

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=",")
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])
        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.aug_transform = videomae_video_transforms.create_random_augment(
            input_size=(self.input_size, self.input_size),
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )
        self.eval_transform = videomae_video_transforms.Compose(
            [
                videomae_video_transforms.Resize(self.short_side_size, interpolation="bilinear"),
                videomae_video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
                volume_transforms.ClipToTensor(),
                videomae_video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def _load_video_with_indices(self, sample_path: str) -> tuple[np.ndarray, list[int]]:
        if not os.path.exists(sample_path):
            raise FileNotFoundError(sample_path)
        vr = VideoReader(sample_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Empty video: {sample_path}")

        # Segment-based sampling (SSV2-style): used for SSV2, KTH, and similar datasets
        _segment_datasets = {
            "ssv2", "ssv2_5k", "ssv2_chiral", "haa100", "penn", "single_object",
            "kth", "kth-5", "kth-2", "penn-action",
        }
        if self.data_set.lower() in _segment_datasets:
            average_duration = total_frames // self.num_frames
            if average_duration > 0:
                if self.view_mode == "center_uniform" or self.deterministic:
                    offsets = np.full(self.num_frames, average_duration // 2, dtype=np.int64)
                else:
                    offsets = np.random.randint(average_duration, size=self.num_frames)
                frame_indices = (np.multiply(list(range(self.num_frames)), average_duration) + offsets).astype(np.int64).tolist()
            elif total_frames > self.num_frames:
                if self.view_mode == "center_uniform" or self.deterministic:
                    frame_indices = np.linspace(0, total_frames - 1, self.num_frames)
                    frame_indices = np.rint(frame_indices).astype(np.int64).tolist()
                else:
                    frame_indices = np.sort(np.random.randint(total_frames, size=self.num_frames)).astype(np.int64).tolist()
            else:
                frame_indices = list(range(total_frames))
                while len(frame_indices) < self.num_frames:
                    frame_indices.append(frame_indices[-1])
        else:
            converted_len = self.num_frames * self.sampling_rate
            if total_frames <= converted_len:
                frame_indices = np.linspace(0, max(total_frames - 1, 0), self.num_frames)
                frame_indices = np.rint(frame_indices).astype(np.int64).tolist()
            else:
                start_idx = max(0, (total_frames - converted_len) // 2)
                end_idx = start_idx + converted_len
                frame_indices = np.linspace(start_idx, end_idx - 1, self.num_frames)
                frame_indices = np.rint(frame_indices).astype(np.int64).tolist()

        frames = vr.get_batch(frame_indices).asnumpy()
        return frames, frame_indices

    def _sample_spatial_params(self, height: int, width: int) -> tuple[int, int, int, int]:
        return videomae_video_transforms._get_param_spatial_crop(  # type: ignore[attr-defined]
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333),
            height=height,
            width=width,
        )

    def _transform_frames(self, frames: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.view_mode == "center_uniform" or self.deterministic:
            h, w = frames.shape[1], frames.shape[2]
            video = self.eval_transform(list(frames))
            crop_params = _compute_resize_and_center_crop_params(
                height=h,
                width=w,
                short_side_size=self.short_side_size,
                input_size=self.input_size,
            )
        else:
            # Training augmentation: Normalize -> RandomCrop -> Resize
            frame_list = [Image.fromarray(frame) for frame in frames]
            frame_list = self.aug_transform(frame_list)
            frame_tensors = [self.rgb_transform(frame) for frame in frame_list]
            video = torch.stack(frame_tensors, dim=0)  # T,C,H,W
            video = video.permute(0, 2, 3, 1)  # T,H,W,C
            video = _tensor_normalize(video, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            video = video.permute(3, 0, 1, 2)  # C,T,H,W
            crop_params = self._sample_spatial_params(video.shape[2], video.shape[3])
            video = _apply_crop_and_resize(video, crop_params, self.input_size)
        return video, {"crop_params": crop_params}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        sample = self.dataset_samples[index]
        sample_path = _resolve_sample_path(self.data_root, sample)
        frames, frame_indices = self._load_video_with_indices(sample_path)
        video, spatial_meta = self._transform_frames(frames)
        meta = {
            "sample_id": _build_sample_id(sample),
            "video_path": sample_path,
            "frame_indices": frame_indices,
            **spatial_meta,
        }
        return video, int(self.label_array[index]), meta


class LocalConceptVideoDataset(LocalVideoDataset):
    def __init__(
        self,
        anno_path: str | Path,
        data_root: str | Path,
        data_set: str,
        pseudo_mask_root: str | Path,
        tubelet_size: int,
        patch_size: int,
        target_cache_root: str | Path | None = None,
        num_frames: int = 16,
        sampling_rate: int = 4,
        input_size: int = 224,
        deterministic: bool = True,
        view_mode: str = "random",
        predownsampled: bool = False,
    ) -> None:
        super().__init__(
            anno_path=anno_path,
            data_root=data_root,
            data_set=data_set,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            input_size=input_size,
            deterministic=deterministic,
            view_mode=view_mode,
        )
        self.pseudo_mask_root = Path(pseudo_mask_root)
        self.tubelet_size = int(tubelet_size)
        self.patch_size = int(patch_size)
        self.target_cache_root = Path(target_cache_root) if target_cache_root is not None else None
        self.predownsampled = bool(predownsampled)

    def _target_cache_path(self, meta: dict[str, Any]) -> Path | None:
        if self.target_cache_root is None:
            return None
        key_payload = {
            "frame_indices": [int(idx) for idx in meta["frame_indices"]],
            "crop_params": [int(v) for v in meta["crop_params"]],
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "tubelet_size": self.tubelet_size,
            "view_mode": self.view_mode,
        }
        digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.target_cache_root / meta["sample_id"] / f"{digest}.pt"

    def _load_predownsampled(self, sample_id: str) -> torch.Tensor:
        """Load a mask that was already downsampled by build_pseudo_labels.py."""
        mask_path = self.pseudo_mask_root / sample_id / "pixel_mask.npy"
        mask = np.load(mask_path)
        target = torch.from_numpy(mask).float()
        if target.ndim == 3:
            target = target.unsqueeze(0)
        return target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor, dict[str, Any]]:
        video, label, meta = super().__getitem__(index)
        if self.predownsampled:
            target = self._load_predownsampled(meta["sample_id"])
        else:
            cache_path = self._target_cache_path(meta)
            if cache_path is not None and cache_path.exists():
                target = torch.load(cache_path, map_location="cpu", weights_only=True)
            else:
                target = build_target_from_meta(
                    meta=meta,
                    mask_root=self.pseudo_mask_root,
                    tubelet_size=self.tubelet_size,
                    input_size=self.input_size,
                    patch_size=self.patch_size,
                )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(target.contiguous().to(dtype=torch.uint8), cache_path)
        target = target.to(dtype=torch.float32)
        return video, label, target, meta

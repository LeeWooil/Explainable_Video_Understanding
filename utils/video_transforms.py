from __future__ import annotations

import math
import numbers
import random
from collections.abc import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _pil_interp(method: str):
    if method == "bicubic":
        return Image.BICUBIC
    if method == "lanczos":
        return Image.LANCZOS
    if method == "hamming":
        return Image.HAMMING
    return Image.BILINEAR


def _get_param_spatial_crop(
    scale: tuple[float, float],
    ratio: tuple[float, float],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = math.exp(random.uniform(*log_ratio))
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            return top, left, crop_h, crop_w

    in_ratio = width / height
    if in_ratio < ratio[0]:
        crop_w = width
        crop_h = int(round(crop_w / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_h = height
        crop_w = int(round(crop_h * ratio[1]))
    else:
        crop_h = height
        crop_w = width
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return top, left, crop_h, crop_w


class _FrameListRandAugment:
    def __init__(self, input_size: tuple[int, int], interpolation: str = "bicubic") -> None:
        resize = transforms.Resize(input_size, interpolation=_pil_interp(interpolation))
        self.transform = transforms.Compose([resize, transforms.RandAugment()])

    def __call__(self, frames: Sequence[Image.Image]) -> list[Image.Image]:
        return [self.transform(frame) for frame in frames]


def create_random_augment(
    input_size: tuple[int, int],
    auto_augment: str = "rand-m7-n4-mstd0.5-inc1",
    interpolation: str = "bicubic",
):
    del auto_augment
    return _FrameListRandAugment(input_size=input_size, interpolation=interpolation)


class Compose:
    def __init__(self, transforms_list: Sequence) -> None:
        self.transforms = list(transforms_list)

    def __call__(self, clip):
        for transform in self.transforms:
            clip = transform(clip)
        return clip


class Resize:
    """Resize a list of HWC numpy frames while preserving aspect ratio."""

    def __init__(self, size, interpolation: str = "nearest") -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not clip:
            return []
        if isinstance(self.size, numbers.Number):
            target = int(self.size)
            h, w = clip[0].shape[:2]
            if (w <= h and w == target) or (h <= w and h == target):
                new_h, new_w = h, w
            elif w < h:
                new_w = target
                new_h = int(target * h / w)
            else:
                new_h = target
                new_w = int(target * w / h)
        else:
            new_h, new_w = self.size
        interpolation = cv2.INTER_LINEAR if self.interpolation == "bilinear" else cv2.INTER_NEAREST
        return [cv2.resize(frame, (new_w, new_h), interpolation=interpolation) for frame in clip]


class CenterCrop:
    """Center crop a list of HWC numpy frames."""

    def __init__(self, size) -> None:
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        self.size = tuple(size)

    def __call__(self, clip: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not clip:
            return []
        crop_h, crop_w = self.size
        im_h, im_w = clip[0].shape[:2]
        if crop_w > im_w or crop_h > im_h:
            raise ValueError(
                f"Initial image size ({im_w}, {im_h}) must be >= crop size ({crop_w}, {crop_h})"
            )
        left = int(round((im_w - crop_w) / 2.0))
        top = int(round((im_h - crop_h) / 2.0))
        return [frame[top : top + crop_h, left : left + crop_w, :] for frame in clip]


class Normalize:
    """Normalize a clip tensor of shape C,T,H,W."""

    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=clip.dtype, device=clip.device).view(-1, 1, 1, 1)
        std = torch.tensor(self.std, dtype=clip.dtype, device=clip.device).view(-1, 1, 1, 1)
        return (clip - mean) / std

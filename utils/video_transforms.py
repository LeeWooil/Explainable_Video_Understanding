from __future__ import annotations

import math
import random
from collections.abc import Sequence

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

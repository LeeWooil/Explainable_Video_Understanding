from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image


def _convert_img(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img


class ClipToTensor:
    """Convert a list of HWC frames into a float tensor with shape C,T,H,W."""

    def __init__(self, channel_nb: int = 3, div_255: bool = True, numpy: bool = False) -> None:
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip: Sequence[np.ndarray | Image.Image]):
        if not clip:
            raise ValueError("ClipToTensor received an empty clip")

        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            if ch != self.channel_nb:
                raise AssertionError(f"Got {ch} instead of {self.channel_nb} channels")
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(f"Expected numpy.ndarray or PIL.Image but got {type(clip[0])}")

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])
        for img_idx, img in enumerate(clip):
            if isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            elif not isinstance(img, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray or PIL.Image but got {type(img)}")
            np_clip[:, img_idx, :, :] = _convert_img(img)

        if self.div_255:
            np_clip = np_clip / 255.0
        if self.numpy:
            return np_clip

        tensor_clip = torch.from_numpy(np_clip)
        if not isinstance(tensor_clip, torch.FloatTensor):
            tensor_clip = tensor_clip.float()
        return tensor_clip

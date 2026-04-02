from __future__ import annotations

from pathlib import Path

import numpy as np


def bilinear_sample(flow_frame: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width, _ = flow_frame.shape
    x = np.clip(x, 0.0, width - 1.0)
    y = np.clip(y, 0.0, height - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    wa = np.where((x0 == x1) & (y0 == y1), 1.0, wa)
    wb = np.where((x0 == x1) & (y0 != y1), 0.0, wb)
    wc = np.where((x0 != x1) & (y0 == y1), 0.0, wc)
    wd = np.where((x0 == x1) | (y0 == y1), 0.0, wd)

    top_left = flow_frame[y0, x0]
    bottom_left = flow_frame[y1, x0]
    top_right = flow_frame[y0, x1]
    bottom_right = flow_frame[y1, x1]
    return (
        top_left * wa[..., None]
        + bottom_left * wb[..., None]
        + top_right * wc[..., None]
        + bottom_right * wd[..., None]
    )


def pad_flow_to_length(flow: np.ndarray, length: int, pad_mode: str) -> tuple[np.ndarray, int]:
    num_frames = int(flow.shape[0])
    if num_frames >= length:
        return flow, 0
    if num_frames <= 0:
        raise ValueError("Cannot pad an empty flow sequence.")

    pad_count = length - num_frames
    if pad_mode == "edge":
        pad = np.repeat(flow[-1:], pad_count, axis=0)
    elif pad_mode == "zero":
        pad = np.zeros((pad_count, *flow.shape[1:]), dtype=flow.dtype)
    elif pad_mode == "reflect":
        if num_frames == 1:
            pad = np.repeat(flow[-1:], pad_count, axis=0)
        else:
            indices = []
            while len(indices) < pad_count:
                indices.extend(range(num_frames - 2, -1, -1))
                indices.extend(range(1, num_frames))
            pad = flow[np.asarray(indices[:pad_count], dtype=np.int32)]
    else:
        raise ValueError(f"Unsupported pad mode: {pad_mode}")
    return np.concatenate([flow, pad], axis=0), pad_count


def compute_segment_start_indices(num_frames: int, length: int, num_segments: int) -> np.ndarray:
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    effective_num_frames = max(int(num_frames), int(length))
    segment_edges = np.linspace(0.0, float(effective_num_frames), num_segments + 1)
    segment_centers = 0.5 * (segment_edges[:-1] + segment_edges[1:])
    start_indices = np.rint(segment_centers - (length / 2.0)).astype(np.int32)
    return np.clip(start_indices, 0, effective_num_frames - length)


def build_trajectory_stacks(flow: np.ndarray, length: int, num_segments: int | None = None, pad_mode: str = "edge") -> np.ndarray:
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape [T, H, W, 2], got {flow.shape}")

    flow, _ = pad_flow_to_length(flow, length, pad_mode)
    num_frames, height, width, _ = flow.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    if num_segments is None:
        start_indices = np.arange(num_frames - length + 1, dtype=np.int32)
    else:
        start_indices = compute_segment_start_indices(num_frames, length, num_segments)

    outputs = []
    for start in start_indices:
        pos_x = xx.copy()
        pos_y = yy.copy()
        steps = []
        for step in range(length):
            sampled = bilinear_sample(flow[start + step], pos_x, pos_y)
            steps.append(sampled)
            if step < length - 1:
                pos_x = pos_x + sampled[..., 0]
                pos_y = pos_y + sampled[..., 1]
        outputs.append(np.stack(steps, axis=0))

    return np.stack(outputs, axis=0).astype(np.float32)


def reconstruct_absolute_trajectories(displacements: np.ndarray, coords: np.ndarray) -> np.ndarray:
    displacements = np.asarray(displacements, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.int32)
    if displacements.ndim != 3 or displacements.shape[-1] != 2:
        raise ValueError(f"Expected displacements with shape [N, T, 2], got {displacements.shape}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords with shape [N, 3], got {coords.shape}")
    if displacements.shape[0] != coords.shape[0]:
        raise ValueError("displacements and coords must have the same first dimension")

    num_trajectories, length, _ = displacements.shape
    absolute = np.zeros((num_trajectories, length, 2), dtype=np.float32)
    absolute[:, 0, 0] = coords[:, 2].astype(np.float32)
    absolute[:, 0, 1] = coords[:, 1].astype(np.float32)
    if length > 1:
        cumulative = np.cumsum(displacements[:, :-1, :], axis=1, dtype=np.float32)
        absolute[:, 1:, :] = absolute[:, :1, :] + cumulative
    return absolute


def build_grouped_pixel_masks(
    trajectories: np.ndarray,
    coords: np.ndarray,
    height: int,
    width: int,
    temporal_mode: str,
    start_frames: np.ndarray,
    total_frames: int | None,
    labels: np.ndarray | None = None,
    num_groups: int | None = None,
    preserve_label_ids: bool = False,
) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.int32)
    start_frames = np.asarray(start_frames, dtype=np.int32)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError(f"Expected trajectories with shape [N,T,2], got {trajectories.shape}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords with shape [N,3], got {coords.shape}")
    if len(trajectories) != len(coords) or len(trajectories) != len(start_frames):
        raise ValueError("trajectories, coords, and start_frames must have the same first dimension")

    absolute = reconstruct_absolute_trajectories(trajectories, coords)
    num_trajectories, length, _ = absolute.shape

    if labels is None:
        labels = np.zeros(num_trajectories, dtype=np.int32)
    else:
        labels = np.asarray(labels, dtype=np.int32)
        if len(labels) != num_trajectories:
            raise ValueError("labels must match the number of trajectories")
        valid_mask = labels >= 0
        if not np.any(valid_mask):
            raise RuntimeError("No valid cluster labels available for mask generation.")
        absolute = absolute[valid_mask]
        start_frames = start_frames[valid_mask]
        labels = labels[valid_mask]
        if not preserve_label_ids:
            unique_labels = np.unique(labels)
            remap = {int(label): idx for idx, label in enumerate(unique_labels.tolist())}
            labels = np.asarray([remap[int(label)] for label in labels.tolist()], dtype=np.int32)

    inferred_num_groups = int(labels.max()) + 1 if len(labels) else 1
    if num_groups is None:
        num_groups = inferred_num_groups
    elif inferred_num_groups > int(num_groups):
        raise ValueError(f"labels require at least {inferred_num_groups} groups, but num_groups={num_groups}")

    if temporal_mode == "relative":
        time_length = length
    elif temporal_mode == "absolute":
        time_length = int(total_frames) if total_frames is not None else int(start_frames.max()) + length
    else:
        raise ValueError(f"Unsupported temporal_mode: {temporal_mode}")

    masks = np.zeros((int(num_groups), time_length, height, width), dtype=np.uint8)
    xs = np.rint(absolute[:, :, 0]).astype(np.int32)
    ys = np.rint(absolute[:, :, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    base_frames = np.arange(length, dtype=np.int32)[None, :]
    if temporal_mode == "relative":
        frame_indices = np.broadcast_to(base_frames, xs.shape)
    else:
        frame_indices = start_frames[:, None] + base_frames
        valid &= (frame_indices >= 0) & (frame_indices < time_length)

    if np.any(valid):
        group_ids = np.broadcast_to(labels[:, None], xs.shape)
        masks[group_ids[valid], frame_indices[valid], ys[valid], xs[valid]] = 1
    return masks


def load_trajectory_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
    elif path.suffix == ".npz":
        obj = np.load(path)
        if "trajectory" not in obj:
            raise KeyError(f"Expected key 'trajectory' in {path}")
        arr = obj["trajectory"]
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")
    if arr.ndim != 5 or arr.shape[-1] != 2:
        raise ValueError(f"Expected [K,L,H,W,2], got {arr.shape}")
    return arr


def flat_indices_to_coords(flat_indices: np.ndarray, height: int, width: int) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    hw = height * width
    segment_ids = flat_indices // hw
    rem = flat_indices % hw
    ys = rem // width
    xs = rem % width
    return np.stack([segment_ids, ys, xs], axis=1).astype(np.int32)


def compute_path_lengths_from_stacks(trajectory_stacks: np.ndarray) -> np.ndarray:
    if trajectory_stacks.shape[1] < 2:
        return np.zeros(trajectory_stacks.shape[0] * trajectory_stacks.shape[2] * trajectory_stacks.shape[3], dtype=np.float32)
    safe_stacks = np.asarray(trajectory_stacks, dtype=np.float32)
    step_vectors = np.diff(safe_stacks, axis=1)
    step_lengths = np.linalg.norm(step_vectors, axis=-1)
    return step_lengths.sum(axis=1).reshape(-1).astype(np.float32)


def build_motion_filter_info(
    path_lengths: np.ndarray,
    motion_threshold: float | None,
    motion_threshold_percentile: float | None,
) -> tuple[np.ndarray, dict]:
    threshold_source = "disabled"
    if motion_threshold_percentile is not None:
        threshold = float(np.percentile(path_lengths, motion_threshold_percentile))
        threshold_source = f"percentile_{motion_threshold_percentile:g}"
    elif motion_threshold is not None:
        threshold = float(motion_threshold)
        threshold_source = "absolute"
    else:
        threshold = None

    stats = {
        "path_length_min": float(path_lengths.min()) if len(path_lengths) else 0.0,
        "path_length_median": float(np.median(path_lengths)) if len(path_lengths) else 0.0,
        "path_length_mean": float(path_lengths.mean()) if len(path_lengths) else 0.0,
        "path_length_max": float(path_lengths.max()) if len(path_lengths) else 0.0,
    }
    if threshold is None:
        return np.ones(len(path_lengths), dtype=bool), {"motion_filter": {"enabled": False, "threshold_source": threshold_source, **stats}}

    selected_mask = path_lengths >= threshold
    if not np.any(selected_mask):
        raise RuntimeError("Motion thresholding removed all trajectories. Lower --motion-threshold or percentile.")
    return selected_mask, {
        "motion_filter": {
            "enabled": True,
            "threshold_source": threshold_source,
            "threshold_value": float(threshold),
            "num_selected_after_motion": int(np.sum(selected_mask)),
            "num_filtered_out_by_motion": int(len(selected_mask) - np.sum(selected_mask)),
            **stats,
        }
    }


def compute_segment_start_frames(
    num_segments: int,
    trajectory_length: int,
    trajectory_start_mode: str,
    num_flow_frames: int | None,
) -> np.ndarray:
    if trajectory_start_mode == "sliding":
        return np.arange(num_segments, dtype=np.int32)
    if trajectory_start_mode == "segment":
        if num_flow_frames is None:
            raise ValueError("--num-flow-frames is required when --trajectory-start-mode=segment.")
        return compute_segment_start_indices(num_flow_frames, trajectory_length, num_segments).astype(np.int32)
    raise ValueError(f"Unsupported trajectory_start_mode: {trajectory_start_mode}")


def load_saliency_mask(mask_root: Path, sample_id: str, frame_idx: int, expected_hw: tuple[int, int]) -> np.ndarray:
    mask_path = mask_root / sample_id / f"frame_{frame_idx:06d}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Saliency mask not found: {mask_path}")
    mask = np.asarray(np.load(mask_path))
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D saliency mask at {mask_path}, got shape {mask.shape}")
    if tuple(mask.shape) != expected_hw:
        raise ValueError(f"Saliency mask shape mismatch at {mask_path}: expected {expected_hw}, got {mask.shape}")
    return mask

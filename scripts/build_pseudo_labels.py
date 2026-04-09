from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.trajectory_ops import (
    build_grouped_pixel_masks,
    flat_indices_to_coords,
    load_trajectory_array,
)
from utils.pseudo_labels import (
    select_mask_by_frame_indices,
    apply_spatial_transform_to_mask,
    spatial_pool_to_patches,
    pool_mask_to_tubelets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build absolute-time VideoMAE pseudo labels from global clustering results.")
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument(
        "--global-result-dir",
        type=Path,
        required=True,
        help="Global clustering result directory containing selected_sample_ids/selected_trajectory_indices/cluster_labels.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--raw-flow-root",
        type=Path,
        default=None,
        help="Optional raw-flow root used to infer num_flow_frames when trajectory files lack that info.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)

    # --- Downsampling: replicate training-time spatial/temporal pooling ---
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Spatial max-pool stride (e.g. 16). Matches spatial_pool_to_patches at training time.",
    )
    parser.add_argument(
        "--tubelet-size",
        type=int,
        default=None,
        help="Temporal max-pool stride (e.g. 2). Matches pool_mask_to_tubelets at training time.",
    )
    # Video access args (required when --patch-size / --tubelet-size is set)
    parser.add_argument(
        "--anno-path",
        type=Path,
        nargs="+",
        default=None,
        help="Annotation CSV(s) mapping video paths to labels (same format as training).",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory for video files.")
    parser.add_argument("--data-set", type=str, default=None, help="Dataset name (e.g. SSv2_chiral, kth).")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to sample.")
    parser.add_argument("--sampling-rate", type=int, default=4, help="Sampling rate for non-segment datasets.")
    parser.add_argument("--input-size", type=int, default=224, help="Spatial input size after resize/crop.")
    parser.add_argument("--short-side-size", type=int, default=224, help="Short side size for center_uniform resize.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training-equivalent downsampling
# ---------------------------------------------------------------------------

def _build_sample_id(sample: str) -> str:
    """Same logic as local_video_dataset._build_sample_id."""
    return Path(sample).with_suffix("").as_posix()


def _load_annotation_samples(anno_paths: list[Path]) -> dict[str, str]:
    """Load annotation CSVs and return {sample_id: relative_video_path}."""
    result: dict[str, str] = {}
    for path in anno_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                sample = parts[0].strip()
                result[_build_sample_id(sample)] = sample
    return result


def _precompute_video_metadata(
    sample_ids: list[str],
    anno_samples: dict[str, str],
    data_root: Path,
) -> dict[str, tuple[int, int, int]]:
    """Return {sample_id: (total_frames, height, width)} for each sample."""
    from decord import VideoReader, cpu as decord_cpu

    metadata: dict[str, tuple[int, int, int]] = {}
    missing: list[str] = []

    for sid in tqdm(sample_ids, desc="Reading video metadata", unit="video", file=sys.stdout):
        if sid not in anno_samples:
            missing.append(sid)
            continue
        sample = anno_samples[sid]
        video_path = Path(sample)
        if not video_path.is_absolute():
            video_path = data_root / sample
        if not video_path.exists():
            missing.append(sid)
            continue
        try:
            vr = VideoReader(str(video_path), num_threads=1, ctx=decord_cpu(0))
            total_frames = len(vr)
            frame0 = vr[0].asnumpy()
            h, w = frame0.shape[0], frame0.shape[1]
            metadata[sid] = (total_frames, h, w)
            del vr
        except Exception as e:
            print(f"[warn] Failed to read video for {sid}: {e}", flush=True)
            missing.append(sid)

    if missing:
        print(
            f"[warn] Could not get video metadata for {len(missing)} / {len(sample_ids)} samples.",
            flush=True,
        )
    return metadata


def _compute_deterministic_frame_indices(
    total_video_frames: int,
    num_frames: int,
    sampling_rate: int,
    data_set: str,
) -> list[int]:
    """Replicate LocalVideoDataset._load_video_with_indices (deterministic / center_uniform)."""
    _segment_datasets = {
        "ssv2", "ssv2_5k", "ssv2_chiral", "haa100", "penn", "single_object",
        "kth", "kth-5", "kth-2", "penn-action",
    }
    if data_set.lower() in _segment_datasets:
        average_duration = total_video_frames // num_frames
        if average_duration > 0:
            offsets = np.full(num_frames, average_duration // 2, dtype=np.int64)
            frame_indices = (
                np.multiply(list(range(num_frames)), average_duration) + offsets
            ).astype(np.int64).tolist()
        elif total_video_frames > num_frames:
            frame_indices = np.rint(
                np.linspace(0, total_video_frames - 1, num_frames)
            ).astype(np.int64).tolist()
        else:
            frame_indices = list(range(total_video_frames))
            while len(frame_indices) < num_frames:
                frame_indices.append(frame_indices[-1])
    else:
        converted_len = num_frames * sampling_rate
        if total_video_frames <= converted_len:
            frame_indices = np.rint(
                np.linspace(0, max(total_video_frames - 1, 0), num_frames)
            ).astype(np.int64).tolist()
        else:
            start_idx = max(0, (total_video_frames - converted_len) // 2)
            end_idx = start_idx + converted_len
            frame_indices = np.rint(
                np.linspace(start_idx, end_idx - 1, num_frames)
            ).astype(np.int64).tolist()
    return frame_indices


def _compute_center_crop_params(
    video_h: int,
    video_w: int,
    short_side_size: int,
    input_size: int,
) -> tuple[int, int, int, int]:
    """Replicate LocalVideoDataset._transform_frames center_uniform crop logic."""
    h, w = video_h, video_w
    if (w <= h and w == short_side_size) or (h <= w and h == short_side_size):
        new_h, new_w = h, w
    elif w < h:
        new_w = short_side_size
        new_h = int(short_side_size * h / w)
    else:
        new_h = short_side_size
        new_w = int(short_side_size * w / h)
    crop_h = crop_w = input_size
    top = max((new_h - crop_h) // 2, 0)
    left = max((new_w - crop_w) // 2, 0)
    return (top, left, crop_h, crop_w)


def _downsample_mask_like_training(
    mask: np.ndarray,
    total_video_frames: int,
    video_h: int,
    video_w: int,
    data_set: str,
    num_frames: int,
    sampling_rate: int,
    short_side_size: int,
    input_size: int,
    patch_size: int,
    tubelet_size: int,
) -> np.ndarray:
    """Apply the exact same pipeline as build_target_from_meta at training time.

    Pipeline (matches pseudo_labels.py + local_video_dataset.py center_uniform):
      1. select_mask_by_frame_indices  — deterministic segment-based sampling
      2. spatial crop                  — center_uniform crop params
      3. apply_spatial_transform       — nearest-resize to (input_size, input_size)
      4. spatial_pool_to_patches       — max_pool2d(kernel=patch_size)
      5. pool_mask_to_tubelets         — temporal max-pool(tubelet_size)
    """
    # 1. Frame selection (deterministic / center_uniform)
    frame_indices = _compute_deterministic_frame_indices(
        total_video_frames, num_frames, sampling_rate, data_set,
    )
    selected = select_mask_by_frame_indices(mask, frame_indices)

    # 2. Spatial crop (center_uniform)
    crop_params = _compute_center_crop_params(video_h, video_w, short_side_size, input_size)
    top, left, crop_h, crop_w = crop_params
    if selected.ndim == 3:
        cropped = selected[:, top : top + crop_h, left : left + crop_w]
    else:
        cropped = selected[:, :, top : top + crop_h, left : left + crop_w]

    t = torch.from_numpy(np.array(cropped, copy=True)).float()

    # 3. Resize to input_size (nearest interpolation)
    resized = apply_spatial_transform_to_mask(
        t, crop_params=(0, 0, t.shape[-2], t.shape[-1]), input_size=input_size,
    )

    # 4. Spatial pool to patches
    patch_mask = spatial_pool_to_patches(resized, patch_size=patch_size)

    # 5. Temporal pool to tubelets
    pooled = pool_mask_to_tubelets(patch_mask, tubelet_size=tubelet_size)
    if pooled.ndim == 3:
        pooled = pooled.unsqueeze(0)

    return pooled.numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Coverage stats & global result loading (unchanged)
# ---------------------------------------------------------------------------

def _compute_coverage_stats(mask: np.ndarray) -> dict:
    if mask.ndim == 4:
        concept_occupancy = mask.reshape(mask.shape[0], -1).sum(axis=1)
        frame_active = mask.any(axis=0).reshape(mask.shape[1], -1).sum(axis=1)
        return {
            "num_concepts": int(mask.shape[0]),
            "num_frames": int(mask.shape[1]),
            "active_frame_ratio": float((frame_active > 0).mean()) if len(frame_active) else 0.0,
            "mean_active_patches": float(frame_active.mean()) if len(frame_active) else 0.0,
            "max_active_patches": int(frame_active.max()) if len(frame_active) else 0,
            "mean_active_patches_per_concept": float(concept_occupancy.mean()) if len(concept_occupancy) else 0.0,
            "active_concept_ratio": float((concept_occupancy > 0).mean()) if len(concept_occupancy) else 0.0,
        }
    active_per_frame = mask.reshape(mask.shape[0], -1).sum(axis=1)
    return {
        "num_frames": int(mask.shape[0]),
        "active_frame_ratio": float((active_per_frame > 0).mean()) if len(active_per_frame) else 0.0,
        "mean_active_patches": float(active_per_frame.mean()) if len(active_per_frame) else 0.0,
        "max_active_patches": int(active_per_frame.max()) if len(active_per_frame) else 0,
    }


def load_global_result(global_result_dir: Path) -> dict[str, object]:
    sample_ids_path = global_result_dir / "selected_sample_ids.npy"
    sample_indices_path = global_result_dir / "selected_sample_indices.npy"
    sample_index_to_id_path = global_result_dir / "sample_index_to_id.json"

    if sample_ids_path.exists():
        selected_sample_ids = np.load(sample_ids_path).astype(str)
    else:
        if not sample_indices_path.exists():
            raise FileNotFoundError(
                "Missing required global clustering files in "
                f"{global_result_dir}: {sample_ids_path} or {sample_indices_path}"
            )
        if not sample_index_to_id_path.exists():
            raise FileNotFoundError(
                "Missing required global clustering files in "
                f"{global_result_dir}: {sample_index_to_id_path}"
            )

        selected_sample_indices = np.load(sample_indices_path).astype(np.int32)
        with open(sample_index_to_id_path, "r", encoding="utf-8") as f:
            sample_index_to_id = json.load(f)
        selected_sample_ids = np.asarray(
            [str(sample_index_to_id[str(int(sample_idx))]) for sample_idx in selected_sample_indices.tolist()],
            dtype="<U256",
        )

    selected_trajectory_indices = np.load(global_result_dir / "selected_trajectory_indices.npy").astype(np.int32)
    cluster_labels = np.load(global_result_dir / "cluster_labels.npy").astype(np.int32)
    selected_start_frames = np.load(global_result_dir / "selected_start_frames.npy").astype(np.int32)
    with open(global_result_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    if not (
        len(selected_sample_ids)
        == len(selected_trajectory_indices)
        == len(cluster_labels)
        == len(selected_start_frames)
    ):
        raise ValueError(
            f"Global result length mismatch in {global_result_dir}: "
            f"sample_ids={len(selected_sample_ids)}, "
            f"trajectory_indices={len(selected_trajectory_indices)}, "
            f"cluster_labels={len(cluster_labels)}, "
            f"start_frames={len(selected_start_frames)}"
        )

    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"trajectory_indices": [], "cluster_labels": [], "start_frames": []}
    )
    for sample_id, trajectory_idx, cluster_label, start_frame in zip(
        selected_sample_ids.tolist(),
        selected_trajectory_indices.tolist(),
        cluster_labels.tolist(),
        selected_start_frames.tolist(),
    ):
        grouped[str(sample_id)]["trajectory_indices"].append(int(trajectory_idx))
        grouped[str(sample_id)]["cluster_labels"].append(int(cluster_label))
        grouped[str(sample_id)]["start_frames"].append(int(start_frame))

    grouped_arrays = {
        sample_id: {
            "trajectory_indices": np.asarray(payload["trajectory_indices"], dtype=np.int32),
            "cluster_labels": np.asarray(payload["cluster_labels"], dtype=np.int32),
            "start_frames": np.asarray(payload["start_frames"], dtype=np.int32),
        }
        for sample_id, payload in grouped.items()
    }
    valid_labels = cluster_labels[cluster_labels >= 0]
    num_concepts = int(valid_labels.max()) + 1 if len(valid_labels) else 0
    return {
        "by_sample": grouped_arrays,
        "num_concepts": num_concepts,
        "summary": summary,
        "global_result_dir": str(global_result_dir),
    }


def _stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _resolve_raw_flow_path(sample_id: str, args: argparse.Namespace) -> Path:
    cached_path = args._raw_flow_path_cache.get(sample_id)
    if cached_path is not None:
        return Path(cached_path)

    candidate_roots = []
    if args.raw_flow_root is not None:
        candidate_roots.append(args.raw_flow_root)

    for root in candidate_roots:
        candidate = root / f"{sample_id}.npy"
        if candidate.exists():
            args._raw_flow_path_cache[sample_id] = str(candidate)
            return candidate
        split_candidates = [
            root / "train" / f"{sample_id}.npy",
            root / "val" / f"{sample_id}.npy",
        ]
        existing_split_candidates = [path for path in split_candidates if path.exists()]
        if len(existing_split_candidates) == 1:
            args._raw_flow_path_cache[sample_id] = str(existing_split_candidates[0])
            return existing_split_candidates[0]
        if len(existing_split_candidates) > 1:
            print(
                f"[warn] Multiple raw flow candidates found for sample_id={sample_id}: "
                f"{existing_split_candidates}. Using the first one."
            )
            args._raw_flow_path_cache[sample_id] = str(existing_split_candidates[0])
            return existing_split_candidates[0]

    raise FileNotFoundError(
        f"Could not infer num_flow_frames for sample_id={sample_id}. "
        "Provide --raw-flow-root that contains the matching raw flow."
    )


def _infer_num_flow_frames(sample_id: str, args: argparse.Namespace) -> int:
    cached_value = args._num_flow_frames_cache.get(sample_id)
    if cached_value is not None:
        return int(cached_value)

    flow = np.load(_resolve_raw_flow_path(sample_id, args), mmap_mode="r")
    num_flow_frames = int(flow.shape[0])
    args._num_flow_frames_cache[sample_id] = num_flow_frames
    return num_flow_frames


def _build_global_sample(
    sample_id: str,
    sample_payload: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, int, int, int, int, int]:
    input_path = args.trajectory_root / f"{sample_id}.npy"
    if not input_path.exists():
        raise FileNotFoundError(f"Trajectory file not found for sample_id={sample_id}: {input_path}")

    trajectory_stacks = load_trajectory_array(input_path)
    num_flow_frames = _infer_num_flow_frames(sample_id, args)
    flat_indices = sample_payload["trajectory_indices"]
    cluster_labels = sample_payload["cluster_labels"]
    start_frames = sample_payload["start_frames"]

    _, trajectory_length, height, width, _ = trajectory_stacks.shape
    coords = flat_indices_to_coords(flat_indices, height=height, width=width)
    trajectories = np.asarray(
        trajectory_stacks[coords[:, 0], :, coords[:, 1], coords[:, 2], :],
        dtype=np.float32,
    )
    extra_info = {
        "source": "global_result_dir",
        "global_result_dir": str(args.global_result_dir),
        "num_selected_from_global_result": int(len(flat_indices)),
        "num_noise_labels": int(np.sum(cluster_labels < 0)),
    }
    return (
        trajectories,
        coords,
        start_frames,
        cluster_labels,
        extra_info,
        num_flow_frames,
        _stable_int(sample_id),
        int(height),
        int(width),
        int(trajectory_length),
    )


def process_global_sample(sample_id: str, sample_payload: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    (
        trajectories,
        coords,
        start_frames,
        cluster_labels,
        extra_info,
        num_flow_frames,
        seed_value,
        height,
        width,
        trajectory_length,
    ) = _build_global_sample(sample_id=sample_id, sample_payload=sample_payload, args=args)
    if len(trajectories) == 0:
        return

    pixel_mask = build_grouped_pixel_masks(
        trajectories=trajectories,
        coords=coords,
        height=height,
        width=width,
        temporal_mode="absolute",
        start_frames=start_frames,
        total_frames=num_flow_frames,
        labels=cluster_labels,
        num_groups=args.num_concepts,
        preserve_label_ids=True,
    )

    raw_shape = pixel_mask.shape
    video_meta = args._video_metadata.get(sample_id) if args._video_metadata else None

    if video_meta is not None and args.patch_size is not None and args.tubelet_size is not None:
        total_video_frames, video_h, video_w = video_meta
        pixel_mask = _downsample_mask_like_training(
            pixel_mask,
            total_video_frames=total_video_frames,
            video_h=video_h,
            video_w=video_w,
            data_set=args.data_set,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            short_side_size=args.short_side_size,
            input_size=args.input_size,
            patch_size=args.patch_size,
            tubelet_size=args.tubelet_size,
        )

    save_dir = args.output_root / sample_id
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pixel_mask.npy", pixel_mask.astype(np.uint8))
    metadata = {
        "sample_id": sample_id,
        "num_flow_frames": int(num_flow_frames),
        "trajectory_length": int(trajectory_length),
        "pixel_height": int(height),
        "pixel_width": int(width),
        "num_surviving_trajectories": int(len(trajectories)),
        "num_concepts": int(args.num_concepts),
        "num_labeled_trajectories": int(np.sum(cluster_labels >= 0)),
        "coverage": _compute_coverage_stats(pixel_mask),
        "preprocessing": extra_info,
        "seed_hint": int(seed_value),
        "downsampling": {
            "patch_size": args.patch_size,
            "tubelet_size": args.tubelet_size,
            "raw_shape": list(int(d) for d in raw_shape),
            "stored_shape": list(int(d) for d in pixel_mask.shape),
        },
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _process_global_sample_task(task: tuple[str, dict[str, np.ndarray], argparse.Namespace]) -> None:
    sample_id, sample_payload, args = task
    process_global_sample(sample_id, sample_payload, args)


def _run_tasks_in_parallel(
    tasks: list[tuple],
    task_fn,
    num_workers: int,
    desc: str,
    unit: str,
) -> None:
    if num_workers <= 1:
        for task in tqdm(tasks, desc=desc, unit=unit, file=sys.stdout, dynamic_ncols=True):
            task_fn(task)
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(task_fn, tasks, chunksize=1)
        for _ in tqdm(iterator, total=len(tasks), desc=desc, unit=unit, file=sys.stdout, dynamic_ncols=True):
            pass


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    args._num_flow_frames_cache = {}
    args._raw_flow_path_cache = {}

    global_result = load_global_result(args.global_result_dir)
    args.num_concepts = int(global_result["num_concepts"])
    sample_ids = sorted(global_result["by_sample"].keys())
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    # Pre-compute video metadata when downsampling is requested
    downsample_requested = args.patch_size is not None or args.tubelet_size is not None
    if downsample_requested:
        if args.anno_path is None or args.data_root is None or args.data_set is None:
            raise ValueError(
                "--anno-path, --data-root, and --data-set are required "
                "when --patch-size or --tubelet-size is specified."
            )
        anno_samples = _load_annotation_samples(args.anno_path)
        args._video_metadata = _precompute_video_metadata(sample_ids, anno_samples, args.data_root)
        print(
            f"Video metadata loaded for {len(args._video_metadata)} / {len(sample_ids)} samples. "
            f"Downsampling: patch_size={args.patch_size}, tubelet_size={args.tubelet_size}.",
            flush=True,
        )
    else:
        args._video_metadata = {}

    print(
        f"Loaded global result with {len(global_result['by_sample'])} samples "
        f"across {args.num_concepts} concepts from {args.global_result_dir}",
        flush=True,
    )
    print(
        f"Building pseudo labels for {len(sample_ids)} samples "
        f"with {args.num_workers} worker(s).",
        flush=True,
    )

    tasks = [(sample_id, global_result["by_sample"][sample_id], args) for sample_id in sample_ids]
    _run_tasks_in_parallel(
        tasks=tasks,
        task_fn=_process_global_sample_task,
        num_workers=args.num_workers,
        desc="Building pseudo labels",
        unit="sample",
    )
    print(f"Finished building pseudo labels for {len(sample_ids)} samples.", flush=True)


if __name__ == "__main__":
    main()

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
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.trajectory_ops import (
    build_motion_filter_info,
    build_grouped_pixel_masks,
    build_trajectory_stacks,
    compute_segment_start_frames,
    compute_path_lengths_from_stacks,
    flat_indices_to_coords,
    load_saliency_mask,
    load_trajectory_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build absolute-time VideoMAE pseudo labels from flow trajectories.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--flow-root", type=Path)
    input_group.add_argument("--trajectory-root", type=Path)
    parser.add_argument(
        "--global-result-dir",
        type=Path,
        default=None,
        help="Optional global clustering result directory containing selected_sample_ids/selected_trajectory_indices/cluster_labels.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trajectory-length", type=int, default=16)
    parser.add_argument("--trajectory-stride", type=int, default=4)
    parser.add_argument("--trajectory-start-mode", type=str, default="sliding", choices=("sliding", "segment"))
    parser.add_argument(
        "--raw-flow-root",
        type=Path,
        default=None,
        help="Optional raw-flow root used to infer num_flow_frames when --trajectory-root is used.",
    )
    parser.add_argument("--motion-threshold", type=float, default=None)
    parser.add_argument("--motion-threshold-percentile", type=float, default=None)
    parser.add_argument("--saliency-mask-root", type=Path, default=None)
    parser.add_argument("--saliency-filter-mode", type=str, default=None, choices=("start_frame",))
    parser.add_argument(
        "--concept-map-path",
        type=Path,
        default=None,
        help="Optional sequence_cluster_map.json mapping sample_id[trajectory=<flat_idx>] to global concept id.",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-trajectory-cache", action="store_true")
    args = parser.parse_args()
    if args.global_result_dir is not None and args.trajectory_root is None:
        raise ValueError("--global-result-dir requires --trajectory-root.")
    if args.global_result_dir is not None and args.concept_map_path is not None:
        raise ValueError("Use either --global-result-dir or --concept-map-path, not both.")
    return args


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


def load_concept_map(concept_map_path: Path) -> tuple[dict[str, int], int]:
    with open(concept_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected concept map JSON object at {concept_map_path}")
    concept_map = {str(key): int(value) for key, value in raw.items()}
    num_concepts = max(concept_map.values()) + 1 if concept_map else 0
    return concept_map, num_concepts


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


def _relative_sample_key(flow_root: Path, flow_path: Path) -> str:
    relative_path = flow_path.relative_to(flow_root)
    return str(relative_path.with_suffix(""))


def _build_sliding_stacks(flow: np.ndarray, length: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    full_stacks = build_trajectory_stacks(flow, length=length, num_segments=None)
    start_frames = np.arange(full_stacks.shape[0], dtype=np.int32)[::stride]
    return full_stacks[::stride].astype(np.float32), start_frames


def _ensure_runtime_caches(args: argparse.Namespace) -> None:
    if not hasattr(args, "_num_flow_frames_cache"):
        args._num_flow_frames_cache = {}
    if not hasattr(args, "_raw_flow_path_cache"):
        args._raw_flow_path_cache = {}


def _resolve_raw_flow_path(sample_id: str, args: argparse.Namespace) -> Path:
    _ensure_runtime_caches(args)
    cached_path = args._raw_flow_path_cache.get(sample_id)
    if cached_path is not None:
        return Path(cached_path)

    candidate_roots = []
    if args.raw_flow_root is not None:
        candidate_roots.append(args.raw_flow_root)
    if args.flow_root is not None:
        candidate_roots.append(args.flow_root)

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
        "Provide --raw-flow-root or --flow-root that contains the matching raw flow."
    )


def _infer_num_flow_frames(sample_id: str, args: argparse.Namespace) -> int:
    _ensure_runtime_caches(args)
    cached_value = args._num_flow_frames_cache.get(sample_id)
    if cached_value is not None:
        return int(cached_value)

    flow = np.load(_resolve_raw_flow_path(sample_id, args), mmap_mode="r")
    num_flow_frames = int(flow.shape[0])
    args._num_flow_frames_cache[sample_id] = num_flow_frames
    return num_flow_frames


def _load_input_stacks(input_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    if args.flow_root is not None:
        flow = np.load(input_path).astype(np.float32)
        stacks, start_frames = _build_sliding_stacks(
            flow=flow,
            length=args.trajectory_length,
            stride=args.trajectory_stride,
        )
        return stacks, start_frames, int(flow.shape[0])

    trajectory_stacks = load_trajectory_array(input_path)
    sample_id = _relative_sample_key(args.trajectory_root, input_path)
    num_flow_frames = _infer_num_flow_frames(sample_id, args)
    start_frames = compute_segment_start_frames(
        num_segments=trajectory_stacks.shape[0],
        trajectory_length=trajectory_stacks.shape[1],
        trajectory_start_mode=args.trajectory_start_mode,
        num_flow_frames=num_flow_frames,
    ).astype(np.int32)
    return trajectory_stacks, start_frames, num_flow_frames


def _filter_trajectory_stacks(
    trajectory_stacks: np.ndarray,
    start_frames: np.ndarray,
    sample_id: str,
    saliency_sample_id: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    num_segments, _, height, width, _ = trajectory_stacks.shape
    total_trajectories = num_segments * height * width
    selected_mask = np.ones((num_segments, height, width), dtype=bool)
    extra_info: dict = {
        "trajectory_start_mode": "sliding",
        "num_flow_frames": int(args.num_flow_frames),
    }

    if args.saliency_filter_mode == "start_frame":
        saliency_selected_mask = np.zeros_like(selected_mask)
        for frame_idx in np.unique(start_frames).tolist():
            mask = load_saliency_mask(
                args.saliency_mask_root,
                saliency_sample_id,
                int(frame_idx),
                expected_hw=(height, width),
            )
            segment_ids = np.flatnonzero(start_frames == frame_idx)
            if len(segment_ids) > 0:
                saliency_selected_mask[segment_ids] = mask[None, :, :] == 1
        if not np.any(saliency_selected_mask):
            raise RuntimeError(f"Saliency filtering removed all trajectories for sample_id={sample_id}.")
        selected_mask &= saliency_selected_mask
        extra_info.update(
            {
                "saliency_mask_root": str(args.saliency_mask_root),
                "saliency_filter_mode": "start_frame",
                "num_selected_after_saliency": int(np.sum(saliency_selected_mask)),
                "num_filtered_out_by_saliency": int(total_trajectories - np.sum(saliency_selected_mask)),
            }
        )

    path_lengths = compute_path_lengths_from_stacks(trajectory_stacks).reshape(num_segments, height, width)
    motion_reference_lengths = path_lengths[selected_mask]
    motion_selected_subset, motion_info = build_motion_filter_info(
        path_lengths=motion_reference_lengths,
        motion_threshold=args.motion_threshold,
        motion_threshold_percentile=args.motion_threshold_percentile,
    )
    motion_selected_mask = np.zeros_like(selected_mask)
    motion_selected_mask[selected_mask] = motion_selected_subset
    selected_mask &= motion_selected_mask
    extra_info.update(motion_info)

    kept_indices = np.flatnonzero(selected_mask.reshape(-1)).astype(np.int32)
    coords = flat_indices_to_coords(kept_indices, height=height, width=width)
    kept_start_frames = start_frames[coords[:, 0]]
    trajectories = np.asarray(
        trajectory_stacks[coords[:, 0], :, coords[:, 1], coords[:, 2], :],
        dtype=np.float32,
    )
    return trajectories, coords, kept_start_frames, kept_indices, extra_info


def _build_concept_labels_for_sample(
    sample_id: str,
    kept_indices: np.ndarray,
    concept_map: dict[str, int],
) -> np.ndarray:
    concept_labels = np.full(len(kept_indices), -1, dtype=np.int32)
    for idx, flat_idx in enumerate(kept_indices.tolist()):
        key = f"{sample_id}[trajectory={int(flat_idx)}]"
        if key in concept_map:
            concept_labels[idx] = int(concept_map[key])
    return concept_labels


def _stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


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


def process_flow_file(flow_path: Path, args: argparse.Namespace) -> None:
    trajectory_stacks, start_frames, num_flow_frames = _load_input_stacks(flow_path, args)
    root = args.flow_root if args.flow_root is not None else args.trajectory_root
    sample_id = _relative_sample_key(root, flow_path)
    saliency_sample_id = flow_path.stem
    if trajectory_stacks.size == 0:
        return

    args.num_flow_frames = int(num_flow_frames)
    trajectories, coords, kept_start_frames, kept_indices, extra_info = _filter_trajectory_stacks(
        trajectory_stacks=trajectory_stacks,
        start_frames=start_frames,
        sample_id=sample_id,
        saliency_sample_id=saliency_sample_id,
        args=args,
    )
    if len(trajectories) == 0:
        return

    _, _, height, width, _ = trajectory_stacks.shape
    concept_labels = None
    labeled_trajectory_count = None
    if args.concept_map is not None:
        concept_labels = _build_concept_labels_for_sample(
            sample_id=sample_id,
            kept_indices=kept_indices,
            concept_map=args.concept_map,
        )
        labeled_trajectory_count = int(np.sum(concept_labels >= 0))
        if labeled_trajectory_count == 0:
            return

    pixel_mask = build_grouped_pixel_masks(
        trajectories=trajectories,
        coords=coords,
        height=height,
        width=width,
        temporal_mode="absolute",
        start_frames=kept_start_frames,
        total_frames=num_flow_frames,
        labels=concept_labels,
        num_groups=args.num_concepts if concept_labels is not None else None,
        preserve_label_ids=concept_labels is not None,
    )
    if concept_labels is None:
        pixel_mask = pixel_mask[0]

    save_dir = args.output_root / sample_id
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pixel_mask.npy", pixel_mask.astype(np.uint8))
    metadata = {
        "sample_id": sample_id,
        "num_flow_frames": int(num_flow_frames),
        "trajectory_length": int(trajectory_stacks.shape[1]),
        "trajectory_stride": int(args.trajectory_stride) if args.flow_root is not None else None,
        "trajectory_start_mode": args.trajectory_start_mode,
        "pixel_height": int(height),
        "pixel_width": int(width),
        "num_surviving_trajectories": int(len(trajectories)),
        "coverage": _compute_coverage_stats(pixel_mask),
        "preprocessing": extra_info,
    }
    if concept_labels is not None:
        metadata["num_concepts"] = int(args.num_concepts)
        metadata["num_labeled_trajectories"] = int(labeled_trajectory_count)
        metadata["concept_map_path"] = str(args.concept_map_path)
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


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

    save_dir = args.output_root / sample_id
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "pixel_mask.npy", pixel_mask.astype(np.uint8))
    metadata = {
        "sample_id": sample_id,
        "num_flow_frames": int(num_flow_frames),
        "trajectory_length": int(trajectory_length),
        "trajectory_stride": None,
        "trajectory_start_mode": args.trajectory_start_mode,
        "pixel_height": int(height),
        "pixel_width": int(width),
        "num_surviving_trajectories": int(len(trajectories)),
        "num_concepts": int(args.num_concepts),
        "num_labeled_trajectories": int(np.sum(cluster_labels >= 0)),
        "coverage": _compute_coverage_stats(pixel_mask),
        "preprocessing": extra_info,
        "seed_hint": int(seed_value),
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _process_global_sample_task(task: tuple[str, dict[str, np.ndarray], argparse.Namespace]) -> None:
    sample_id, sample_payload, args = task
    process_global_sample(sample_id, sample_payload, args)


def _process_flow_file_task(task: tuple[Path, argparse.Namespace]) -> None:
    flow_path, args = task
    process_flow_file(flow_path, args)


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
    args.concept_map = None
    args.num_concepts = 0
    args.global_result = None
    _ensure_runtime_caches(args)
    if args.global_result_dir is not None:
        args.global_result = load_global_result(args.global_result_dir)
        args.num_concepts = int(args.global_result["num_concepts"])
        sample_ids = sorted(args.global_result["by_sample"].keys())
        if args.limit is not None:
            sample_ids = sample_ids[: args.limit]
        print(
            f"Loaded global result with {len(args.global_result['by_sample'])} samples "
            f"across {args.num_concepts} concepts from {args.global_result_dir}",
            flush=True,
        )
        print(
            f"Building pseudo labels for {len(sample_ids)} samples from global result "
            f"with {args.num_workers} worker(s).",
            flush=True,
        )
        tasks = [(sample_id, args.global_result["by_sample"][sample_id], args) for sample_id in sample_ids]
        _run_tasks_in_parallel(
            tasks=tasks,
            task_fn=_process_global_sample_task,
            num_workers=args.num_workers,
            desc="Building pseudo labels",
            unit="sample",
        )
        print(f"Finished building pseudo labels for {len(sample_ids)} samples.", flush=True)
        return
    elif args.concept_map_path is not None:
        args.concept_map, args.num_concepts = load_concept_map(args.concept_map_path)
        print(
            f"Loaded concept map with {len(args.concept_map)} trajectory assignments "
            f"across {args.num_concepts} concepts from {args.concept_map_path}",
            flush=True,
        )
    root = args.flow_root if args.flow_root is not None else args.trajectory_root
    flow_paths = sorted(root.rglob("*.npy"))
    if args.limit is not None:
        flow_paths = flow_paths[: args.limit]
    print(f"Found {len(flow_paths)} input files under {root}; using {args.num_workers} worker(s).", flush=True)
    tasks = [(flow_path, args) for flow_path in flow_paths]
    _run_tasks_in_parallel(
        tasks=tasks,
        task_fn=_process_flow_file_task,
        num_workers=args.num_workers,
        desc="Building pseudo labels",
        unit="sample",
    )
    print(f"Finished building pseudo labels for {len(flow_paths)} files.", flush=True)


if __name__ == "__main__":
    main()

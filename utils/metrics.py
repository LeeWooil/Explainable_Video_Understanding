from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist


def compute_binary_localization_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, float]:
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}")

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    tp = float(((preds == 1.0) & (targets == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (targets == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (targets == 1.0)).sum().item())
    tn = float(((preds == 0.0) & (targets == 0.0)).sum().item())
    total = tp + fp + fn + tn

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (total + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pred_positive_ratio": float(preds.mean().item()),
        "target_positive_ratio": float(targets.mean().item()),
    }


class RunningBinaryLocalizationMetrics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.num_batches = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.count = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor, threshold: float = 0.5) -> None:
        stats = compute_binary_localization_stats(logits=logits, targets=targets, threshold=threshold)
        self.loss_sum += float(loss.item())
        self.num_batches += 1
        self.tp += stats["tp"]
        self.fp += stats["fp"]
        self.fn += stats["fn"]
        self.tn += stats["tn"]
        numel = float(targets.numel())
        self.pred_sum += stats["pred_positive_ratio"] * numel
        self.target_sum += stats["target_positive_ratio"] * numel
        self.count += numel

    def compute(self, eps: float = 1e-8) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        return {
            "loss": self.loss_sum / max(self.num_batches, 1),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "pred_positive_ratio": self.pred_sum / max(self.count, 1.0),
            "target_positive_ratio": self.target_sum / max(self.count, 1.0),
        }

    def synchronize_between_processes(self, device: torch.device) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        state = torch.tensor(
            [
                self.loss_sum,
                float(self.num_batches),
                self.tp,
                self.fp,
                self.fn,
                self.tn,
                self.pred_sum,
                self.target_sum,
                self.count,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
        self.loss_sum = float(state[0].item())
        self.num_batches = int(state[1].item())
        self.tp = float(state[2].item())
        self.fp = float(state[3].item())
        self.fn = float(state[4].item())
        self.tn = float(state[5].item())
        self.pred_sum = float(state[6].item())
        self.target_sum = float(state[7].item())
        self.count = float(state[8].item())

"""Shared evaluation utilities — chunk-level + recording-level accuracy.

Model-agnostic: provide a forward_fn(model, batch) -> logits so this works
with HuggingFace models and plain torch modules alike.

Recording-level aggregation softmaxes each chunk, averages the probability
vectors across all chunks of a recording, then argmaxes.
"""

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

ForwardFn = Callable[[torch.nn.Module, dict], torch.Tensor]


def hf_forward(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """Default forward: HuggingFace audio-classification convention."""
    return model(input_values=batch["input_values"]).logits


def torch_forward(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """Plain nn.Module forward: model(x) returns logits directly."""
    return model(batch["input_values"])


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: ForwardFn = hf_forward,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (softmax probs, labels) aligned with the loader's order."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        logits = forward_fn(model, batch_dev)
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def aggregate_by_recording(
    probs: np.ndarray, labels: np.ndarray, recording_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean softmax across a recording's chunks, then argmax → one prediction per file."""
    by_rec: dict[str, list[np.ndarray]] = defaultdict(list)
    rec_label: dict[str, int] = {}
    for p, y, rid in zip(probs, labels, recording_ids):
        by_rec[rid].append(p)
        rec_label[rid] = int(y)
    rec_ids = list(by_rec.keys())
    rec_preds = np.array([np.mean(by_rec[r], axis=0).argmax() for r in rec_ids])
    rec_labels = np.array([rec_label[r] for r in rec_ids])
    return rec_preds, rec_labels


def report_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    recording_ids: list[str],
    class_names: list[str],
) -> dict:
    """Print chunk- and recording-level metrics and return the two accuracies."""
    chunk_preds = probs.argmax(axis=1)
    chunk_acc = accuracy_score(labels, chunk_preds)
    print("\n=== Chunk-level ===")
    print(f"Accuracy: {chunk_acc:.4f}")
    print(classification_report(labels, chunk_preds, target_names=class_names))

    rec_preds, rec_labels = aggregate_by_recording(probs, labels, recording_ids)
    rec_acc = accuracy_score(rec_labels, rec_preds)
    print("=== Recording-level ===")
    print(f"Accuracy: {rec_acc:.4f}")
    print(classification_report(rec_labels, rec_preds, target_names=class_names))
    return {"chunk_acc": float(chunk_acc), "record_acc": float(rec_acc)}

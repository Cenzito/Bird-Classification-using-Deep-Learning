"""Random search over ResNet18 hyperparameters (batch size, LR, dropout).

Each trial retrains from scratch with a sampled (bs, lr, dropout) combo,
following the same two-phase schedule as train.py but for fewer epochs.
Saves the full results table (CSV) and the best config (JSON).
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# Shared code/ modules + reuse train.py's transform and inner loop.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import BirdSongChunkDataset, load_splits  # noqa: E402
from evaluate import predict, torch_forward  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import build_transform, train_one_epoch  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
SPLITS_CSV = Path(__file__).resolve().parents[2] / "data" / "splits_10birds.csv"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "short_audio_10"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "resnet18"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Search space + trial budget ───────────────────────────────────────
BATCH_SIZES = [16, 32, 64]
LR_RANGE = (1e-4, 1e-2)       # sampled log-uniformly
DROPOUT_RANGE = (0.1, 0.6)
N_TRIALS = 15
TUNE_PHASE1 = 3               # shorter than train.py — just a ranking signal
TUNE_PHASE2 = 5
SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0


def sample_hparams() -> dict:
    """Draw one random hyperparameter configuration from the search space."""
    bs = random.choice(BATCH_SIZES)
    lr = 10 ** random.uniform(np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1]))
    dropout = random.uniform(*DROPOUT_RANGE)
    return {"batch_size": bs, "lr": round(lr, 6), "dropout": round(dropout, 3)}


def build_model(num_classes: int, dropout: float) -> nn.Module:
    """Fresh pretrained ResNet18 with a configurable dropout on the head."""
    # Local build_model so dropout can vary per trial (train.py's version
    # bakes in the module-level DROPOUT constant).
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def run_trial(
    hp: dict, train_ds, val_ds, num_classes: int
) -> float:
    """Train one random configuration and return best val accuracy."""
    model = build_model(num_classes, hp["dropout"]).to(DEVICE)

    # Rebuild loaders per trial because batch_size is itself a tunable.
    train_loader = DataLoader(
        train_ds,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: head only ────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=hp["lr"])
    for epoch in range(1, TUNE_PHASE1 + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
        print(
            f"    p1 epoch {epoch}/{TUNE_PHASE1} | "
            f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
        )

    # ── Phase 2: full fine-tuning with 0.1× backbone LR ───────────────
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if "fc" not in n],
             "lr": hp["lr"] * 0.1},
            {"params": model.fc.parameters(), "lr": hp["lr"]},
        ]
    )

    # Rank trials by peak val accuracy (not final) — robust to late overfitting.
    best_val_acc = 0.0
    for epoch in range(1, TUNE_PHASE2 + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())
        best_val_acc = max(best_val_acc, val_acc)
        print(
            f"    p2 epoch {epoch}/{TUNE_PHASE2} | "
            f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
        )

    del model
    return best_val_acc


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df, _, class_names = load_splits(SPLITS_CSV)
    transform = build_transform()
    train_ds = BirdSongChunkDataset(df, "train", AUDIO_ROOT, transform)
    val_ds = BirdSongChunkDataset(df, "val", AUDIO_ROOT, transform)
    print(
        f"Classes ({len(class_names)}): {class_names}\n"
        f"Train chunks: {len(train_ds)} | Val: {len(val_ds)}\n"
        f"Device: {DEVICE} | Trials: {N_TRIALS}"
    )

    results: list[dict] = []
    for trial in tqdm(range(1, N_TRIALS + 1), desc="Trials"):
        hp = sample_hparams()
        print(
            f"\n{'='*60}\n"
            f"Trial {trial}/{N_TRIALS} | "
            f"bs={hp['batch_size']} lr={hp['lr']} dropout={hp['dropout']}\n"
            f"{'='*60}"
        )
        best_val_acc = run_trial(hp, train_ds, val_ds, len(class_names))
        hp["best_val_acc"] = best_val_acc
        results.append(hp)

    results_df = (
        pd.DataFrame(results)
        .sort_values("best_val_acc", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n{'='*60}\nRandom Search Results (best → worst)\n{'='*60}")
    print(results_df.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "hparam_search_resnet18.csv"
    results_df.to_csv(csv_path, index=False)

    best = results_df.iloc[0]
    best_hparams = {
        "batch_size": int(best["batch_size"]),
        "lr": float(best["lr"]),
        "dropout": float(best["dropout"]),
        "tuning_val_acc": float(best["best_val_acc"]),
    }
    json_path = OUTPUT_DIR / "best_hparams_resnet18.json"
    with open(json_path, "w") as f:
        json.dump(best_hparams, f, indent=2)

    print(f"\nSaved results table  → {csv_path}")
    print(f"Saved best hparams   → {json_path}")
    print(
        f"Best config: bs={best_hparams['batch_size']} "
        f"lr={best_hparams['lr']} dropout={best_hparams['dropout']} "
        f"→ val_acc={best_hparams['tuning_val_acc']:.4f}"
    )


if __name__ == "__main__":
    main()

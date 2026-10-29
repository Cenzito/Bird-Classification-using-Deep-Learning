"""Train ResNet18 on 10 bird species (mel-spectrogram classification).

Two-phase transfer learning: freeze the pretrained backbone and train the
new head (Phase 1), then unfreeze and fine-tune the full network with a
lower LR on the backbone (Phase 2).
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# Make the shared code/ modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SR, BirdSongChunkDataset, load_splits  # noqa: E402
from evaluate import predict, report_metrics, torch_forward  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
SPLITS_CSV = Path(__file__).resolve().parents[2] / "data" / "splits_10birds.csv"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "short_audio_10"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "resnet18"

# ── Hyperparameters ───────────────────────────────────────────────────
LEARNING_RATE = 1e-3
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
BATCH_SIZE = 32
DROPOUT = 0.3
SEED = 42

# ── Mel-spectrogram settings ──────────────────────────────────────────
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = 224  # ResNet18 input size

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
NUM_WORKERS = 4 if DEVICE.type == "cuda" else 0


def build_transform():
    """Waveform (np.ndarray) → 3×IMG_SIZE×IMG_SIZE tensor for ResNet."""

    def transform(y: np.ndarray) -> torch.Tensor:
        S = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8) # Min-max normalise to [0, 1] per chunk.
        spec = torch.from_numpy(S_db).float().unsqueeze(0).unsqueeze(0)
        spec = F.interpolate(
            spec, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        # Replicate the single channel 3× so the ImageNet stem accepts it.
        return spec.squeeze(0).expand(3, -1, -1).contiguous()

    return transform


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Pretrained ResNet18 with a Dropout + Linear head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Standard supervised loop — returns mean train loss for the epoch."""
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["input_values"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Splits + datasets ──────────────────────────────────────────
    df, _, class_names = load_splits(SPLITS_CSV)
    transform = build_transform()
    train_ds = BirdSongChunkDataset(df, "train", AUDIO_ROOT, transform)
    val_ds = BirdSongChunkDataset(df, "val", AUDIO_ROOT, transform)
    test_ds = BirdSongChunkDataset(df, "test", AUDIO_ROOT, transform)
    print(
        f"Classes ({len(class_names)}): {class_names}\n"
        f"Train chunks: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ── 2. Model + loss ───────────────────────────────────────────────
    model = build_model(len(class_names), freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUTPUT_DIR / "best_resnet18.pt"
    best_val_acc = 0.0

    # ── 3a. Phase 1: train head only (backbone frozen) ────────────────
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    print(f"Phase 1: head-only training ({PHASE1_EPOCHS} epochs)")
    for epoch in tqdm(range(1, PHASE1_EPOCHS + 1), desc="Phase 1"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
        print(f"  p1 epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    # ── 3b. Phase 2: fine-tune full model (lower LR on backbone) ──────
    for p in model.parameters():
        p.requires_grad = True
    # Backbone gets 0.1× LR so pretrained features only get gently updated.
    optimizer = torch.optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if "fc" not in n],
             "lr": LEARNING_RATE * 0.1},
            {"params": model.fc.parameters(), "lr": LEARNING_RATE},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE2_EPOCHS
    )
    print(f"Phase 2: full fine-tuning ({PHASE2_EPOCHS} epochs)")
    for epoch in tqdm(range(1, PHASE2_EPOCHS + 1), desc="Phase 2"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
        scheduler.step()
        print(f"  p2 epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save(model.state_dict(), best_path)
            print(f"    -> saved best model (val_acc={val_acc:.4f})")

    # ── 4. Test metrics on best checkpoint ────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    probs, labels = predict(model, test_loader, DEVICE, torch_forward)
    recording_ids = [s["recording_id"] for s in test_ds.samples]
    report_metrics(probs, labels, recording_ids, class_names)

    # ── 5. Save final weights ─────────────────────────────────────────
    torch.save(model.state_dict(), OUTPUT_DIR / "resnet18.pt")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Saved weights to {OUTPUT_DIR / 'resnet18.pt'}")


if __name__ == "__main__":
    main()

"""Train AlexNet on 10 bird species (mel-spectrogram classification).

Shared data loading + evaluation come from ``code/data.py`` and 
``code/evaluate.py``; this file is only responsible for the model, 
the waveform→spectrogram transform, and the two-phase optimization loop.
"""

import sys
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

# Make the shared code/ modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SR, BirdSongChunkDataset, load_splits  # noqa: E402
from evaluate import predict, report_metrics, torch_forward  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
SPLITS_CSV = Path(__file__).resolve().parents[2] / "data" / "splits_10birds.csv"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "short_audio_10"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "alexnet"

# ── Hyperparameters ───────────────────────────────────────────────────
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15       
PHASE1_EPOCHS = 5
BATCH_SIZE = 128      
NUM_WORKERS = 4
SEED = 42

# ── Mel-spectrogram settings ──────────────────────────────────────────
N_MELS = 128
N_FFT = 2048         
HOP_LENGTH = 512

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def build_transform():
    """Return a callable: waveform (np.ndarray) → 3×H×W tensor for AlexNet.
    
    1. mel-spectrogram via ``librosa.feature.melspectrogram``
    2. log-scale (``librosa.power_to_db``)
    3. normalize to [0, 1]
    4. resize to 224×224 and repeat to 3 channels
    """

    def transform(y: np.ndarray) -> torch.Tensor:
        # 1. Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        
        # 2. Convert to log-scale (decibels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3. Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(mel_spec_norm, dtype=torch.float32)
        
        # 4. Resize to 224x224 and repeat to 3 channels
        # Add batch and channel dimensions for interpolation: (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Remove batch dimension and repeat to 3 channels: (3, 224, 224)
        tensor = tensor.squeeze(0).repeat(3, 1, 1)
        
        return tensor

    return transform


def build_model(num_classes: int) -> nn.Module:
    """Instantiate AlexNet with a new num_classes-way head."""
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    
    # Replace the final linear layer (classifier[6]) with the new num_classes
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
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
    for batch in loader:
        x = batch["input_values"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
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
        drop_last=True,
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

    # ── 2. Model ──────────────────────────────────────────────────────
    model = build_model(len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 3. Train Phase 1 (Head Only) ──────────────────────────────────
    print(f"\nPhase 1: Training classifier head (backbone frozen)")
    print("-" * 55)
    
    # Freeze the entire backbone and intermediate classifier layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the final classification layer
    for param in model.classifier[6].parameters():
        param.requires_grad = True
        
    optimizer1 = torch.optim.Adam(model.classifier[6].parameters(), lr=LEARNING_RATE)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer1, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
        print(f"Epoch {epoch}/{PHASE1_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}")


    # ── 4. Train Phase 2 (Full Model) ─────────────────────────────────
    PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS
    print(f"\nPhase 2: Fine-tuning full model")
    print("-" * 55)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
        
    # Backbone (features) gets a 10x lower learning rate than the head
    optimizer2 = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": LEARNING_RATE * 0.1},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
    ])
    
    # Cosine annealing scheduler for gradual LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer2, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
        scheduler.step()
        
        print(f"Epoch {epoch}/{PHASE2_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_AlexNet.pt")
            print(f" -> Saved best model (val_acc={val_acc:.3f})")

    # Load the best weights we found during training
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_AlexNet.pt"))

    # ── 4. Save ───────────────────────────────────────────────────────
    # We save the final PyTorch model weights (equivalent to trainer.save_model)
    final_model_path = OUTPUT_DIR / "alexnet_final.pt"
    torch.save(model.state_dict(), final_model_path)
    
    print(f"Saved model to {OUTPUT_DIR}")
if __name__ == "__main__":
    main()
"""Train a Sound Event Detection (SED) model on 10 bird species.

Shared data loading + evaluation come from
``code/data.py`` and ``code/evaluate.py``; this file is only responsible for
the SED model (e.g. CNN + attention pooling), the waveform→spectrogram
transform, and the optimization loop.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import torchaudio.transforms as T

# Make the shared code/ modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SR, BirdSongChunkDataset, load_splits  # noqa: E402
from evaluate import predict, report_metrics, torch_forward  # noqa: E402
# ── Paths ─────────────────────────────────────────────────────────────
SPLITS_CSV = Path(__file__).resolve().parents[2] / "data" / "splits_10birds.csv"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "short_audio_10"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "code" / "sed"

# ── Hyperparameters ───────────────────────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2     # AdamW configuration
NUM_EPOCHS = 10         
BATCH_SIZE = 8         
NUM_WORKERS = 0
SEED = 1213

# ── Spectrogram settings ──────────────────────────────────────────────
N_MELS = 128
N_FFT = 2048  
HOP_LENGTH = 512  

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# ── Global Transform Configuration ────────────────────────────────────
MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=32000, 
    n_mels=N_MELS, 
    n_fft=N_FFT, 
    hop_length=HOP_LENGTH, 
    f_min=20,          
    f_max=None,        
    power=2.0,
    normalized=True,
    center=True        
)
AMP_DB = T.AmplitudeToDB(top_db=80.0)

def global_transform(y: np.ndarray) -> torch.Tensor:
    """Waveform (np.ndarray) → N_MELS × T tensor using notebook normalization."""
    y_tensor = torch.from_numpy(y).float()
    if y_tensor.ndim > 1:
        y_tensor = y_tensor.mean(dim=0)
    
    # 1. Mel-spectrogram and log-scale
    spec = MEL_TRANSFORM(y_tensor)
    spec = AMP_DB(spec)
    
    # 2. Add batch/channel dim for per-sample normalization logic
    X = spec.unsqueeze(0) 
    
    # 3. Notebook-exact normalization math
    eps = 1e-6
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)
    
    norm_min = Xstd.min(-1)[0].min(-1)[0]
    norm_max = Xstd.max(-1)[0].max(-1)[0]
    
    V = torch.zeros_like(Xstd)
    fix_ind = (norm_max - norm_min) > eps
    
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(torch.min(V_fix, norm_max_fix), norm_min_fix)
        V[fix_ind] = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
    
    return V.squeeze(0) 



# ── SED Architecture Components ───────────────────────────────────────
class GeMFreq(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), 1)).pow(1.0 / self.p)

class AttHead(nn.Module):
    def __init__(self, in_chans: int, p: float = 0.5, num_class: int = 10):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2), 
            nn.Linear(in_chans, 512), 
            nn.ReLU(), 
            nn.Dropout(p)
        )
        self.attention = nn.Conv1d(in_channels=512, out_channels=num_class, kernel_size=1)
        self.fix_scale = nn.Conv1d(in_channels=512, out_channels=num_class, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        time_att = torch.tanh(self.attention(feat))
        logits = torch.sum(self.fix_scale(feat) * torch.softmax(time_att, dim=-1), dim=-1)
        return logits


class SEDModel(nn.Module):
    """CNN encoder + attention pooling head → clip-level logits."""
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        # ResNet34 Backbone
        self.backbone = timm.create_model("resnet34", features_only=True, pretrained=True, in_chans=1)
        # SED Temporal Attention Head
        self.head = AttHead(self.backbone.feature_info.channels()[-1], num_class=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_MELS, T) → clip-level logits (B, num_classes)."""
        x = x.unsqueeze(1)  
        feats = self.backbone(x)[-1]
        return self.head(feats)


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
        
        y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()
        
        optimizer.zero_grad()
        loss = criterion(model(x), y_one_hot)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Splits + datasets ──────────────────────────────────────────
    df, _, class_names = load_splits(SPLITS_CSV)
    #transform = build_transform()
    train_ds = BirdSongChunkDataset(df, "train", AUDIO_ROOT, global_transform)
    val_ds = BirdSongChunkDataset(df, "val", AUDIO_ROOT, global_transform)
    test_ds = BirdSongChunkDataset(df, "test", AUDIO_ROOT, global_transform)
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

    # ── 2. Model + optimizer ──────────────────────────────────────────
    model = SEDModel(len(class_names)).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    #── 3. Train ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0  # Track the best accuracy!
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_labels = predict(model, val_loader, DEVICE, torch_forward)
        val_acc = (val_probs.argmax(axis=1) == val_labels).mean()
        
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_sed_model.pth")
            print(f"🌟 New best model saved! (val_acc: {best_val_acc:.4f})")


    # ── 4. Test metrics (chunk + recording level) ─────────────────────
    model.eval()
    
    state_dict = torch.load("best_sed_model.pth", map_location=DEVICE)
    keys_to_ignore = ["logmelspec_extractor.0.spectrogram.window", "logmelspec_extractor.0.mel_scale.fb"]
    for key in keys_to_ignore:
        if key in state_dict:
            del state_dict[key]
            print(f"Bypassed: {key}")
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded perfectly!")

    probs, labels = predict(model, test_loader, DEVICE, torch_forward)
    recording_ids = [s["recording_id"] for s in test_ds.samples]
    report_metrics(probs, labels, recording_ids, class_names)

    # ── 5. Save ───────────────────────────────────────────────────────
    torch.save(model.state_dict(), OUTPUT_DIR / "sed_last.pt")
    print(f"Saved final weights to {OUTPUT_DIR / 'sed_last.pt'}")

if __name__ == "__main__":
    main()

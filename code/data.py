"""Shared dataset + split loader for the 10-species BirdCLEF subset.

Model-agnostic: each architecture supplies its own transform callable that
turns a raw waveform (np.ndarray, float32, mono @ 16 kHz) into the tensor the
model expects.

Each sample = one non-overlapping 10-second chunk of a recording.
"""

from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

SR = 16000
CHUNK_DURATION = 10
CHUNK_SAMPLES = SR * CHUNK_DURATION  # 160_000 samples

Transform = Callable[[np.ndarray], torch.Tensor]


def audio_path(audio_root: Path, species: str, filename: str) -> Path:
    """Return the expected path for a recording on disk."""
    return Path(audio_root) / species / filename


def load_splits(csv_path: Path) -> tuple[pd.DataFrame, LabelEncoder, list[str]]:
    """Load splits_10birds.csv and fit a label encoder on the species column."""
    df = pd.read_csv(csv_path)
    encoder = LabelEncoder()
    encoder.fit(sorted(df["primary_label"].unique()))
    df = df.assign(label_id=encoder.transform(df["primary_label"]))
    return df, encoder, list(encoder.classes_)


class BirdSongChunkDataset(Dataset):
    """One sample = one 10-second chunk of a recording.

    Parameters
    ----------
    df
        The full split DataFrame from :func:`load_splits`.
    split
        Which split to materialize: ``"train"`` / ``"val"`` / ``"test"``.
    audio_root
        Directory containing ``<species>/<filename>`` audio files.
    transform
        Callable mapping a mono waveform (np.ndarray, len == CHUNK_SAMPLES)
        to a torch.Tensor — the model's ``input_values``. Architecture-specific
        (e.g. AST feature extractor, mel-spectrogram, raw waveform).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        audio_root: Path,
        transform: Transform,
    ) -> None:
        self.audio_root = Path(audio_root)
        self.transform = transform
        self.samples: list[dict] = []
        subset = df[df["split"] == split].reset_index(drop=True)
        for _, row in subset.iterrows():
            path = audio_path(self.audio_root, row["primary_label"], row["filename"])
            try:
                duration = librosa.get_duration(path=str(path))
            except Exception:
                continue  # skip unreadable files rather than crash a 4000-file scan
            n_chunks = max(1, int(duration) // CHUNK_DURATION)
            for i in range(n_chunks):
                self.samples.append(
                    {
                        "path": str(path),
                        "label_id": int(row["label_id"]),
                        "chunk_idx": i,
                        "recording_id": Path(row["filename"]).stem,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        offset = s["chunk_idx"] * CHUNK_DURATION
        y, _ = librosa.load(
            s["path"], sr=SR, offset=offset, duration=CHUNK_DURATION, mono=True
        )
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)))
        return {
            "input_values": self.transform(y),
            "labels": torch.tensor(s["label_id"], dtype=torch.long),
        }

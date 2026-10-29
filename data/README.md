# BirdCLEF 2021 — 10-Species Subset

Subset of the [BirdCLEF 2021](https://www.kaggle.com/c/birdclef-2021) competition dataset,
containing 10 acoustically diverse species from the top-12 by recording count.

## Directory Structure

```
raw_data/short_audio_10/
├── barswa/                 # Barn Swallow            — 500 recordings
├── comrav/                 # Common Raven            — 500 recordings
├── cubthr/                 # Curve-billed Thrasher   — 500 recordings
├── eursta/                 # European Starling       — 500 recordings
├── gbwwre1/                # Giant Babax Wren        — 500 recordings
├── houspa/                 # House Sparrow           — 500 recordings
├── houwre/                 # House Wren              — 500 recordings
├── norcar/                 # Northern Cardinal       — 500 recordings
├── redcro/                 # Red Crossbill           — 500 recordings
└── rewbla/                 # Red-winged Blackbird    — 500 recordings

data/
├── README.md
├── splits_10birds.csv      # Pre-computed train/val/test split (10 species)
└── train_metadata.csv      # Full competition metadata (all 397 species)
```

**Total:** 5,000 `.ogg` audio files · ~4.4 GB

## `splits_10birds.csv`

Derived from `splits_top12.csv` by filtering to the 10 selected species.
Stratified split: 400 train / 50 val / 50 test per species.

| Column          | Description                       |
| --------------- | --------------------------------- |
| `filename`      | Original `.ogg` filename          |
| `primary_label` | Species code (e.g. `comrav`)      |
| `author`        | Recording author (XenoCanto)      |
| `split`         | `train`, `val`, or `test`         |

### Loading

```python
import pandas as pd
import librosa

df = pd.read_csv("data/splits_10birds.csv")
train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]

AUDIO_DIR = "data/short_audio_10"

# Load a single file
row = train_df.iloc[0]
y, sr = librosa.load(f"{AUDIO_DIR}/{row['primary_label']}/{row['filename']}",
                     sr=22050)
```

Each teammate can decide their own windowing/cropping strategy — the split
assignments are fixed so results are comparable.

See `notebooks/00_load_dataset.ipynb` for visualizations and examples.

## Species

| Code    | Common Name           | Scientific Name                |
| ------- | --------------------- | ------------------------------ |
| barswa  | Barn Swallow          | _Hirundo rustica_              |
| comrav  | Common Raven          | _Corvus corax_                 |
| cubthr  | Curve-billed Thrasher | _Toxostoma curvirostre_        |
| eursta  | European Starling     | _Sturnus vulgaris_             |
| gbwwre1 | Giant Babax Wren      | _Pheugopedius maculipectus_    |
| houspa  | House Sparrow         | _Passer domesticus_            |
| houwre  | House Wren            | _Troglodytes aedon_            |
| norcar  | Northern Cardinal     | _Cardinalis cardinalis_        |
| redcro  | Red Crossbill         | _Loxia curvirostra_            |
| rewbla  | Red-winged Blackbird  | _Agelaius phoeniceus_          |

## Audio Format

- Format: Ogg Vorbis (`.ogg`)
- Duration: 6 s – 2354 s (median ~43 s)
- Sample rate: varies per recording (resample to 22050 Hz on load)

## Source

Original data: Kaggle BirdCLEF 2021 competition.
Full dataset: 397 species · 62,874 recordings · ~42 GB.

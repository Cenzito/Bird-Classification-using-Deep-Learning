"""Fine-tune MIT/ast-finetuned-audioset-10-10-0.4593 on 10 bird species.

Reproduces Config C from the random search (see hyperparam_finetune.py),
which reached 0.95 recording-level test accuracy.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Make the shared code/ modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import SR, BirdSongChunkDataset, load_splits  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
SPLITS_CSV = Path(__file__).resolve().parents[2] / "data" / "splits_10birds.csv"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "short_audio_10"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "ast_winner"

# ── Model ─────────────────────────────────────────────────────────────
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# ── Winner hyperparameters (Config C from the random search) ──────────
LEARNING_RATE = 1.30e-5
WEIGHT_DECAY = 5.09e-2
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EVAL_BATCH_SIZE = 16
WARMUP_STEPS = 50
NUM_WORKERS = 8
SEED = 42


def build_transform(feature_extractor: AutoFeatureExtractor):
    """Wrap the AST feature extractor as a waveform → tensor callable."""

    def transform(y: np.ndarray):
        return feature_extractor(y, sampling_rate=SR, return_tensors="pt")[
            "input_values"
        ].squeeze(0)

    return transform


def compute_metrics(eval_pred) -> dict:
    """Chunk-level accuracy. Recording-level aggregation is in code/evaluate.py."""
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}


def main() -> None:
    set_seed(SEED)

    # ── 1. Splits + datasets ──────────────────────────────────────────
    df, _, class_names = load_splits(SPLITS_CSV)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    transform = build_transform(feature_extractor)
    train_ds = BirdSongChunkDataset(df, "train", AUDIO_ROOT, transform)
    val_ds = BirdSongChunkDataset(df, "val", AUDIO_ROOT, transform)
    print(
        f"Classes ({len(class_names)}): {class_names}\n"
        f"Train chunks: {len(train_ds)} | Val chunks: {len(val_ds)}"
    )

    # ── 2. Model ──────────────────────────────────────────────────────
    id2label = {i: c for i, c in enumerate(class_names)}
    label2id = {c: i for i, c in id2label.items()}
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── 3. Train ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # ── 4. Save ───────────────────────────────────────────────────────
    trainer.save_model(str(OUTPUT_DIR))
    feature_extractor.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved model + feature extractor to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

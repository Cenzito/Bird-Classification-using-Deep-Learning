"""Random log-uniform hyperparameter search + top-3 full-data retrain for AST.

1. Sample N_CONFIGS random configurations on log-uniform scales for
   learning rate and weight decay, and a discrete set for epochs.
2. Train each configuration on a 25 % recording-level subsample of the train
   split, using the full validation split as the ranking metric.
3. Take the top-TOP_K configurations by validation accuracy, retrain each
   on the full train split for FULL_EPOCHS epochs, and evaluate on the
   held-out test split at both chunk- and recording-level.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
from evaluate import aggregate_by_recording  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
SPLITS_CSV = Path(__file__).resolve().parents[2] / "data" / "splits_10birds.csv"
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "short_audio_10"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "ast_hparam"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Model ─────────────────────────────────────────────────────────────
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# ── Sweep settings ────────────────────────────────────────────────────
N_CONFIGS = 16
EPOCH_CHOICES = (3, 5, 7)
SUBSAMPLE_FRAC = 0.25  # fraction of train RECORDINGS kept per sweep run
SUBSAMPLE_SEED = 20260411  # shared across configs → same data for every run
SAMPLE_SEED = 42
TOP_K = 3
FULL_EPOCHS = 5

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


def sample_configs(n: int, rng: np.random.Generator) -> list[dict]:
    """Draw n random configurations on log-uniform LR/WD scales."""
    configs = []
    for run_id in range(n):
        configs.append(
            {
                "run_id": run_id,
                "learning_rate": float(10 ** rng.uniform(-6, -3)),
                "weight_decay": float(10 ** rng.uniform(-5, -1)),
                "num_epochs": int(rng.choice(EPOCH_CHOICES)),
            }
        )
    return configs


def subsample_train_recordings(
    train_df: pd.DataFrame, frac: float, seed: int
) -> pd.DataFrame:
    """Recording-level stratified subsample."""
    recordings = train_df.drop_duplicates("filename")[["filename", "primary_label"]]
    keep, _ = train_test_split(
        recordings,
        train_size=frac,
        stratify=recordings["primary_label"],
        random_state=seed,
    )
    return train_df[train_df["filename"].isin(keep["filename"])].reset_index(drop=True)


def build_model(num_classes: int, id2label: dict) -> AutoModelForAudioClassification:
    """Fresh AST model with a 10-way head; loaded anew for every run."""
    label2id = {v: k for k, v in id2label.items()}
    return AutoModelForAudioClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def make_training_args(cfg: dict, output_dir: Path) -> TrainingArguments:
    """TrainingArguments with LR/WD/epochs pulled from cfg; everything else fixed."""
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=int(cfg["num_epochs"]),
        warmup_steps=WARMUP_STEPS,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        seed=SEED,
    )


def compute_metrics(eval_pred) -> dict:
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}


def best_eval_accuracy(log_history: list[dict]) -> float:
    """Pick the highest ``eval_accuracy`` seen across epochs."""
    accs = [e["eval_accuracy"] for e in log_history if "eval_accuracy" in e]
    return max(accs) if accs else float("nan")


def run_sweep(
    df: pd.DataFrame, class_names: list[str], transform, configs: list[dict]
) -> pd.DataFrame:
    """Train every config on the same 25 % subsample and collect val accuracies."""
    id2label = {i: c for i, c in enumerate(class_names)}
    subsampled_df = pd.concat(
        [
            subsample_train_recordings(
                df[df["split"] == "train"], SUBSAMPLE_FRAC, SUBSAMPLE_SEED
            ),
            df[df["split"] == "val"],
        ],
        ignore_index=True,
    )
    train_ds = BirdSongChunkDataset(subsampled_df, "train", AUDIO_ROOT, transform)
    val_ds = BirdSongChunkDataset(subsampled_df, "val", AUDIO_ROOT, transform)
    print(f"Sweep data: {len(train_ds)} train chunks (25%) | {len(val_ds)} val chunks")

    rows = []
    for cfg in configs:
        print(f"\n── run {cfg['run_id']:02d} | cfg={cfg}")
        model = build_model(len(class_names), id2label)
        trainer = Trainer(
            model=model,
            args=make_training_args(cfg, OUTPUT_DIR / f"sweep_run_{cfg['run_id']:02d}"),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        rows.append(
            {**cfg, "best_val_acc": best_eval_accuracy(trainer.state.log_history)}
        )
        print(f"   best_val_acc={rows[-1]['best_val_acc']:.4f}")
    return (
        pd.DataFrame(rows)
        .sort_values("best_val_acc", ascending=False)
        .reset_index(drop=True)
    )


def retrain_and_test(
    df: pd.DataFrame, class_names: list[str], transform, configs: list[dict]
) -> pd.DataFrame:
    """Retrain each config on the full train split and evaluate on test."""
    id2label = {i: c for i, c in enumerate(class_names)}
    train_ds = BirdSongChunkDataset(df, "train", AUDIO_ROOT, transform)
    val_ds = BirdSongChunkDataset(df, "val", AUDIO_ROOT, transform)
    test_ds = BirdSongChunkDataset(df, "test", AUDIO_ROOT, transform)
    print(
        f"\nFull data: {len(train_ds)} train chunks | "
        f"{len(val_ds)} val | {len(test_ds)} test"
    )

    rows = []
    for cfg in configs:
        full_cfg = {**cfg, "num_epochs": FULL_EPOCHS}
        print(f"\n── retrain run {cfg['run_id']:02d} | cfg={full_cfg}")
        model = build_model(len(class_names), id2label)
        trainer = Trainer(
            model=model,
            args=make_training_args(
                full_cfg, OUTPUT_DIR / f"retrain_run_{cfg['run_id']:02d}"
            ),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        final_val_acc = float(trainer.evaluate(val_ds)["eval_accuracy"])

        # ── Test set: chunk- and recording-level accuracy ─────────────
        pred = trainer.predict(test_ds)
        probs = torch.softmax(torch.from_numpy(pred.predictions), dim=-1).numpy()
        chunk_preds = probs.argmax(axis=1)
        chunk_acc = accuracy_score(pred.label_ids, chunk_preds)

        recording_ids = [s["recording_id"] for s in test_ds.samples]
        rec_preds, rec_labels = aggregate_by_recording(
            probs, pred.label_ids, recording_ids
        )
        rec_acc = accuracy_score(rec_labels, rec_preds)

        rows.append(
            {
                **cfg,
                "final_val_acc": final_val_acc,
                "test_chunk_acc": float(chunk_acc),
                "test_record_acc": float(rec_acc),
            }
        )
        print(
            f"   val={final_val_acc:.4f} | test_chunk={chunk_acc:.4f} "
            f"| test_record={rec_acc:.4f}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(SEED)
    rng = np.random.default_rng(SAMPLE_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, _, class_names = load_splits(SPLITS_CSV)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    transform = build_transform(feature_extractor)

    # ── Phase 1: sweep on 25 % train subsample ────────────────────────
    configs = sample_configs(N_CONFIGS, rng)
    sweep_df = run_sweep(df, class_names, transform, configs)
    sweep_df.to_csv(RESULTS_DIR / "hparam_results.csv", index=False)
    print("\n=== Sweep results (top 5) ===")
    print(sweep_df.head().to_string(index=False))

    # ── Phase 2: top-K retrain on full train, evaluate on test ────────
    top_configs = sweep_df.head(TOP_K).to_dict(orient="records")
    retrain_df = retrain_and_test(df, class_names, transform, top_configs)
    retrain_df.to_csv(RESULTS_DIR / "winner_retrain_results.csv", index=False)

    # ── Final comparison table ────────────────────────────────────────
    comparison = retrain_df.merge(
        sweep_df[["run_id", "best_val_acc"]].rename(
            columns={"best_val_acc": "sweep_val_acc_25pct"}
        ),
        on="run_id",
    )
    print("\n=== Final comparison (sweep val → full val → test) ===")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

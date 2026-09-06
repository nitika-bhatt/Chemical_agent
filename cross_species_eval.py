"""
Cross-Species Generalization Evaluation
========================================
Evaluates the trained HybridARGME ensemble's methylation-site prediction
performance separately on each species in data-11/, and reports both:

  (A) Per-species held-out performance (Accuracy, F1, MCC, AUC)
  (B) A cross-species generalization table, IF you provide per-species
      checkpoints trained with that species excluded (see note below).

──────────────────────────────────────────────────────────────────────────
IMPORTANT METHODOLOGICAL NOTE — read before citing these numbers in your
thesis:

  If `checkpoints/pretrained_esm_v4.pt` was trained on data pooled across
  ALL species (Human + Chimpanzee + Mouse + Rat combined), then running
  this script against that single checkpoint only gives you a per-species
  BREAKDOWN of in-distribution performance — it does NOT demonstrate
  cross-species generalization, because the model has already seen that
  species' distribution during training. That is a meaningful and useful
  number for the thesis (worth reporting), but it should not be labeled
  "generalization."

  To make a genuine generalization claim (train.py splits are per-species
  before combining, per your README), you need a LEAVE-ONE-SPECIES-OUT
  (LOSO) protocol:
    1. Retrain the ensemble N times, each time excluding one species
       entirely from train/val.
    2. Evaluate that excluded-species model ONLY on the held-out species.
    3. Repeat for Human, Chimpanzee, Mouse, Rat.

  This script's `--loso-checkpoint-dir` option supports that: point it at
  a folder containing one checkpoint per held-out species
  (e.g. `loso_checkpoints/without_Mouse.pt`), and it will automatically
  evaluate each checkpoint ONLY on the species it excluded, giving you the
  real generalization numbers for your thesis's "Planned thesis work" item 3.

Usage:
    python cross_species_eval.py \\
        --data-dir data-11 \\
        --checkpoint checkpoints/pretrained_esm_v4.pt \\
        --ptm-subdir Asymmetric_dimethylarginine

    # For the real LOSO generalization table:
    python cross_species_eval.py \\
        --data-dir data-11 \\
        --loso-checkpoint-dir loso_checkpoints \\
        --ptm-subdir Asymmetric_dimethylarginine
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.prediction_agent import HybridARGMEAgent  # noqa: E402


SPECIES = ["Human", "Chimpanzee", "Mouse", "Rat"]


def load_species_windows(data_dir: str, species: str, ptm_subdir: str):
    """
    Loads all 11-residue windows + labels for one species from the
    data-11/<Species>/<PTM_subdir>/ folder structure described in the README.
    Returns (sequences: list[str], labels: list[int]).
    """
    species_dir = os.path.join(data_dir, species, ptm_subdir)
    if not os.path.isdir(species_dir):
        print(f"[WARN] No data directory found for {species}: {species_dir}")
        return [], []

    sequences, labels = [], []

    for csv_path in glob.glob(os.path.join(species_dir, "*positive*.csv")):
        df = pd.read_csv(csv_path)
        col = "Positive_sequence" if "Positive_sequence" in df.columns else df.columns[0]
        sequences.extend(df[col].astype(str).str.upper().tolist())
        labels.extend([1] * len(df))

    for csv_path in glob.glob(os.path.join(species_dir, "*negative*.csv")):
        df = pd.read_csv(csv_path)
        col = "Negative_sequence" if "Negative_sequence" in df.columns else df.columns[0]
        sequences.extend(df[col].astype(str).str.upper().tolist())
        labels.extend([0] * len(df))

    return sequences, labels


def score_windows(agent: HybridARGMEAgent, sequences: list) -> np.ndarray:
    """
    Scores a list of pre-built 11-residue windows directly through the
    loaded ensemble (bypasses predict_sites' R/K candidate scanning, since
    every window here is already centered on a known candidate residue).
    Uses the batched ESM-2 embedding path for speed on large CSVs.
    """
    embeddings = agent.get_esm_embeddings_batched(sequences)

    probs = []
    for emb in embeddings:
        x = emb.unsqueeze(0).to(agent.device)
        fold_probs = []
        for model in agent.models:
            with torch.no_grad():
                logits = model(x)
                fold_probs.append(torch.softmax(logits, dim=1)[:, 1].item())
        probs.append(float(np.mean(fold_probs)))

    return np.array(probs)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "n":           len(labels),
        "accuracy":    accuracy_score(labels, preds),
        "f1":          f1_score(labels, preds, zero_division=0),
        "mcc":         matthews_corrcoef(labels, preds) if len(set(labels)) > 1 else float("nan"),
        "auc":         roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
    }


def run_in_distribution_breakdown(data_dir, checkpoint_path, ptm_subdir):
    """Mode (A): one checkpoint, per-species breakdown."""
    print(f"\nLoading ensemble from {checkpoint_path} ...")
    agent = HybridARGMEAgent(checkpoint_path)

    rows = []
    for species in SPECIES:
        seqs, labels = load_species_windows(data_dir, species, ptm_subdir)
        if not seqs:
            continue
        print(f"[{species}] scoring {len(seqs)} windows...")
        probs = score_windows(agent, seqs)
        metrics = compute_metrics(np.array(labels), probs, agent.avg_threshold)
        metrics["species"] = species
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("species")
    print("\n=== Per-species performance (single combined-training checkpoint) ===")
    print("NOTE: this is an in-distribution breakdown, not a generalization test,")
    print("unless this checkpoint was trained with each species held out.\n")
    print(df.round(4).to_string())
    df.to_csv("cross_species_in_distribution_results.csv")
    print("\nSaved: cross_species_in_distribution_results.csv")


def run_loso(data_dir, loso_checkpoint_dir, ptm_subdir):
    """Mode (B): real leave-one-species-out generalization table."""
    rows = []
    for species in SPECIES:
        ckpt_path = os.path.join(loso_checkpoint_dir, f"without_{species}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing LOSO checkpoint for {species}: {ckpt_path} — skipping.")
            continue

        print(f"\nLoading LOSO checkpoint (trained WITHOUT {species}): {ckpt_path}")
        agent = HybridARGMEAgent(ckpt_path)

        seqs, labels = load_species_windows(data_dir, species, ptm_subdir)
        if not seqs:
            continue
        print(f"[{species}] zero-shot scoring {len(seqs)} windows never seen in training...")
        probs = score_windows(agent, seqs)
        metrics = compute_metrics(np.array(labels), probs, agent.avg_threshold)
        metrics["held_out_species"] = species
        rows.append(metrics)

    if not rows:
        print("\nNo LOSO checkpoints found — nothing to evaluate. "
              "Train one model per species-excluded split first "
              "(see the module docstring).")
        return

    df = pd.DataFrame(rows).set_index("held_out_species")
    print("\n=== TRUE cross-species generalization (leave-one-species-out) ===")
    print(df.round(4).to_string())
    df.to_csv("cross_species_loso_results.csv")
    print("\nSaved: cross_species_loso_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data-11")
    parser.add_argument("--ptm-subdir", default="Asymmetric_dimethylarginine")
    parser.add_argument("--checkpoint", default=None,
                         help="Single combined-training checkpoint (in-distribution breakdown).")
    parser.add_argument("--loso-checkpoint-dir", default=None,
                         help="Folder of without_<Species>.pt checkpoints (real generalization test).")
    args = parser.parse_args()

    if args.loso_checkpoint_dir:
        run_loso(args.data_dir, args.loso_checkpoint_dir, args.ptm_subdir)
    elif args.checkpoint:
        run_in_distribution_breakdown(args.data_dir, args.checkpoint, args.ptm_subdir)
    else:
        default_ckpt = os.path.join("checkpoints", "pretrained_esm_v4.pt")
        print(f"No --checkpoint or --loso-checkpoint-dir given, defaulting to {default_ckpt}")
        run_in_distribution_breakdown(args.data_dir, default_ckpt, args.ptm_subdir)

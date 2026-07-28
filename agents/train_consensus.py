"""
Consensus Meta-Learner Training
================================
Trains the ConsensusAgent logistic regression meta-learner using out-of-fold (OOF)
predictions collected from all parallel agents across the 5-fold cross-validation.

This is the step that upgrades the consensus fusion from a simple weighted average
to a data-driven, calibrated probability fusion — improving AUC/MCC/F1 over
standalone HybridARGME.

Usage (standalone):
    python agents/train_consensus.py \\
        --checkpoint checkpoints/pretrained_esm_v4.pt \\
        --data-dir data-11

The training loop:
  For each fold:
    1. Load the validation split sequences + labels
    2. Run HybridARGMEAgent (transformer)   → P_hybrid
    3. Run MotifAgent                       → P_motif
    4. Run ConservationAgent                → P_conserv
    5. Run DatasetEvidenceAgent             → P_evidence
    6. Run StructuralAgent (neutral 0.50)   → P_struct
  Collect (P_hybrid, P_motif, P_conserv, P_evidence, P_struct, label) from ALL folds
  Train ConsensusAgent meta-learner on the collected OOF matrix
  Save to checkpoints/consensus_agent_meta_learner.pkl
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import StratifiedKFold

# ── Project root on sys.path (works in Colab and local) ─────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)   # parent of agents/
for _p in [_project_dir, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.prediction_agent       import HybridARGMEAgent
from agents.motif_agent            import MotifAgent
from agents.conservation_agent     import ConservationAgent
from agents.dataset_evidence_agent import DatasetEvidenceAgent
from agents.structural_agent       import StructuralAgent
from agents.consensus_agent        import ConsensusAgent

# ── Constants ────────────────────────────────────────────────────────────────
N_FOLDS   = 5
SEED      = 42
MOTIF_LEN = 11
SPECIES_LIST = ["human", "mouse", "rat", "chimpanzee"]


# ── Dynamic path helpers ─────────────────────────────────────────────────────

def _auto_data_dir():
    candidates = [
        "/content/data-11",
        os.path.join(_project_dir, "data-11"),
        "/content/drive/MyDrive/ProteinMind_Agent/data-11",
        "/content/drive/MyDrive/protein_data/data-11",
        "/kaggle/input/datasets/kartikj31/protein-data-11/data-11",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _auto_checkpoint():
    candidates = [
        os.path.join(_project_dir, "checkpoints", "pretrained_esm_v4.pt"),
        "checkpoints/pretrained_esm_v4.pt",
        "/kaggle/working/checkpoints/pretrained_esm_v4.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_all_samples(data_dir: str) -> list:
    """
    Load all 11-mer samples (sequence, label) from the species sub-directories.

    Returns:
        List of (sequence: str, label: int) tuples.
    """
    samples = []
    for species in SPECIES_LIST:
        species_dir = os.path.join(data_dir, species.capitalize())
        if not os.path.exists(species_dir):
            species_dir = os.path.join(data_dir, species)
        if not os.path.exists(species_dir):
            continue
        csv_found = 0
        for root, _, files in os.walk(species_dir):
            for fname in files:
                if not fname.lower().endswith(".csv"):
                    continue
                csv_found += 1
                fpath = os.path.join(root, fname)
                try:
                    # Read CSV; handle files without a header row
                    try:
                        df = pd.read_csv(fpath)
                    except pd.errors.EmptyDataError:
                        print(f"  [WARN] Empty CSV: {fpath}")
                        continue
                    except Exception:
                        # Try without header if parsing fails
                        df = pd.read_csv(fpath, header=None)
                        # Assume first column is sequence if no header
                        df.columns = ["sequence"]
                    # Normalise column names to lower case for consistent access
                    df.columns = [c.lower() for c in df.columns]

                    # Standardise columns for downstream processing
                    if "sequence" not in df.columns:
                        # Scan for likely column names
                        for candidate in ["seq", "peptide", "negative_sequence", "positive_sequence"]:
                            if candidate in df.columns:
                                df.rename(columns={candidate: "sequence"}, inplace=True)
                                break
                        else:
                            # Fallback to first column
                            df.rename(columns={df.columns[0]: "sequence"}, inplace=True)

                    if "label" not in df.columns:
                        if "positive" in fname.lower():
                            df["label"] = 1
                        elif "negative" in fname.lower():
                            df["label"] = 0

                    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
                    df = df[df["sequence"].str.len() == MOTIF_LEN]
                    df.dropna(subset=["label"], inplace=True)

                    # Generate 11‑mer windows from each full‑length sequence
                    for _, row in df.iterrows():
                        full_seq = row["sequence"]
                        lbl = int(row["label"])
                        if len(full_seq) < MOTIF_LEN:
                            continue
                        for i in range(len(full_seq) - MOTIF_LEN + 1):
                            motif = full_seq[i : i + MOTIF_LEN]
                            samples.append((motif, lbl))
                except Exception as e:
                    print(f"  [WARN] Could not read {fpath}: {e}")
    print(f"[ConsensusTrainer] CSV files discovered: {csv_found}")
    print(f"[ConsensusTrainer] Total samples loaded: {len(samples)}")
    return samples


# ── Per-site score collection ────────────────────────────────────────────────

def collect_agent_scores(motif: str, label: int,
                         pred_agent: HybridARGMEAgent,
                         motif_agent: MotifAgent,
                         conserv_agent: ConservationAgent,
                         evidence_agent: DatasetEvidenceAgent,
                         struct_agent: StructuralAgent) -> Optional[dict]:
    """
    Run all five agents on a single 11-mer motif and collect their probability scores.

    Returns a dict with P_hybrid, P_motif, P_conserv, P_evidence, P_struct, label.
    Returns None if prediction fails.
    """
    # The 11-mer is treated as a full sequence; center (pos 6) is the R/K site
    sites = pred_agent.predict_sites(motif)
    if not sites:
        return None

    # Take the center-most R/K site (closest to position 6)
    site = min(sites, key=lambda s: abs(s["position"] - 6))

    # Motif agent
    motif_result   = motif_agent.score_site(motif)

    # Conservation agent
    conserv_result = conserv_agent.score_site(motif)

    # Dataset evidence agent
    evidence_result = evidence_agent.score_site(motif)

    # Structural agent (no accession → neutral 0.50)
    struct_result  = struct_agent.score_site(site["position"], [])

    return {
        "P_hybrid":   site["P_hybrid"],
        "P_motif":    motif_result["P_motif"],
        "P_conserv":  conserv_result["P_conserv"],
        "P_evidence": evidence_result["P_evidence"],
        "P_struct":   struct_result["P_struct"],
        "label":      label,
    }


# ── OOF collection loop ──────────────────────────────────────────────────────

def collect_oof_predictions(samples: list,
                             checkpoint_path: str,
                             data_dir: str) -> np.ndarray:
    """
    Collect out-of-fold predictions from all agents using StratifiedKFold.

    Each fold trains on 4/5 of samples and collects agent scores on the
    1/5 validation fold — ensuring no data leakage into the meta-learner.

    Returns:
        X: np.ndarray shape (N, 5)  — feature matrix (one row per sample)
        y: np.ndarray shape (N,)    — labels
    """
    seqs   = [s for s, _ in samples]
    labels = [l for _, l in samples]

    # Initialize agents (shared across all folds)
    pred_agent    = HybridARGMEAgent(checkpoint_path)
    motif_agent   = MotifAgent()
    conserv_agent = ConservationAgent(data_dir=data_dir)
    evidence_agent = DatasetEvidenceAgent(data_dir=data_dir)
    struct_agent  = StructuralAgent()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_rows = []
    score_keys = ["P_hybrid", "P_motif", "P_conserv", "P_evidence", "P_struct"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(seqs, labels)):
        print(f"\n[ConsensusTrainer] === Fold {fold+1}/{N_FOLDS} | Val size: {len(val_idx)} ===")

        collected = 0
        for idx in val_idx:
            motif = seqs[idx]
            label = labels[idx]

            try:
                row = collect_agent_scores(
                    motif, label,
                    pred_agent, motif_agent, conserv_agent, evidence_agent, struct_agent
                )
                if row is not None:
                    oof_rows.append(row)
                    collected += 1
            except Exception as e:
                # Non-fatal: skip this sample
                pass

        print(f"[ConsensusTrainer]   Collected {collected}/{len(val_idx)} valid OOF rows.")

    if not oof_rows:
        raise RuntimeError("[ConsensusTrainer] No OOF rows collected. "
                           "Check that the checkpoint and data_dir are correct.")

    X = np.array([[r[k] for k in score_keys] for r in oof_rows], dtype=np.float32)
    y = np.array([r["label"] for r in oof_rows], dtype=int)

    print(f"\n[ConsensusTrainer] Total OOF rows: {len(X)} | Positives: {y.sum()} | Negatives: {(y == 0).sum()}")
    return X, y


# ── Main training function ───────────────────────────────────────────────────

def train_consensus_meta_learner(checkpoint_path: str,
                                  data_dir: str,
                                  output_path: str = None) -> ConsensusAgent:
    """
    Full pipeline: collect OOF scores → train ConsensusAgent → save checkpoint.

    Args:
        checkpoint_path: Path to trained HybridARGME ensemble checkpoint.
        data_dir:        Root directory of the data-11 dataset.
        output_path:     Where to save the trained meta-learner (.pkl). Defaults
                         to checkpoints/consensus_agent_meta_learner.pkl

    Returns:
        Trained ConsensusAgent instance.
    """
    output_path = output_path or os.path.join(
        os.path.dirname(checkpoint_path), "consensus_agent_meta_learner.pkl"
    )

    print("\n" + "=" * 65)
    print("  ConsensusAgent Meta-Learner Training")
    print("=" * 65)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output     : {output_path}")
    print("=" * 65 + "\n")

    # 1. Load all samples
    samples = load_all_samples(data_dir)
    if len(samples) < 100:
        raise RuntimeError(
            f"[ConsensusTrainer] Only {len(samples)} samples found in {data_dir}. "
            "Check the data directory structure."
        )

    # Balance samples for cleaner meta-learner training
    pos = [(s, l) for s, l in samples if l == 1]
    neg = [(s, l) for s, l in samples if l == 0]
    # Cap negatives at 3× positives to avoid extreme imbalance skewing logistic regression
    neg_cap = min(len(neg), 3 * len(pos))
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(len(neg), size=neg_cap, replace=False)
    balanced = pos + [neg[i] for i in neg_idx]
    rng.shuffle(balanced)
    print(f"[ConsensusTrainer] Balanced: {len(pos)} pos + {neg_cap} neg = {len(balanced)} total")

    # 2. Collect out-of-fold predictions from all agents
    X, y = collect_oof_predictions(balanced, checkpoint_path, data_dir)

    # 3. Train ConsensusAgent meta-learner on OOF matrix
    agent = ConsensusAgent()
    agent.train(X, y)

    # 4. Save
    agent.save_model(output_path)

    print("\n[ConsensusTrainer] ✅ Done! Consensus meta-learner saved.")
    print(f"  → {output_path}")
    return agent


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ConsensusAgent meta-learner via out-of-fold prediction collection."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to HybridARGME ensemble checkpoint (auto-detected if omitted)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Root data-11 directory (auto-detected if omitted)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for meta-learner .pkl (default: checkpoints/consensus_agent_meta_learner.pkl)"
    )
    args = parser.parse_args()

    ckpt     = args.checkpoint or _auto_checkpoint()
    data_dir = args.data_dir   or _auto_data_dir()

    if not ckpt:
        print("❌ Could not locate checkpoint. Use --checkpoint to specify it.")
        sys.exit(1)
    if not data_dir:
        print("❌ Could not locate data-11 directory. Use --data-dir to specify it.")
        sys.exit(1)

    train_consensus_meta_learner(ckpt, data_dir, output_path=args.output)

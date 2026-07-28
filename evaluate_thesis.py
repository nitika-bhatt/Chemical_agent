import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm

# ── Dynamic project path injection (Colab + local) ──────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from agents.orchestrator import Orchestrator
from model_core import DeepTESite, HybridArgMe


# ── Dynamic path resolution ─────────────────────────────────────────────────

def _auto_data_dir():
    """Auto-detect where data-11 lives (Colab symlink, Drive, or local)."""
    candidates = [
        "/content/data-11",
        os.path.join(_this_dir, "data-11"),
        "/content/drive/MyDrive/ProteinMind_Agent/data-11",
        "/content/drive/MyDrive/protein_data/data-11",
        "/kaggle/input/datasets/kartikj31/protein-data-11/data-11",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _auto_checkpoint():
    """Auto-detect checkpoint path."""
    candidates = [
        "checkpoints/pretrained_esm_v4.pt",
        os.path.join(_this_dir, "checkpoints", "pretrained_esm_v4.pt"),
        "/kaggle/working/checkpoints/pretrained_esm_v4.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _auto_test_csv():
    """Auto-detect a test dataset."""
    candidates = [
        "splits/combined_test.csv",
        os.path.join(_this_dir, "splits", "combined_test.csv"),
        "/kaggle/working/splits/combined_test.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback to tsv in data dir
    d = _auto_data_dir()
    if d:
        tsv = os.path.join(os.path.dirname(d), "data-11", "dataset.tsv")
        if os.path.exists(tsv):
            return tsv
        tsv = os.path.join(d, "dataset.tsv")
        if os.path.exists(tsv):
            return tsv
    return None


# ── Main evaluation function ─────────────────────────────────────────────────

def evaluate_thesis(test_csv_path, checkpoint_path, data_dir=None, num_samples=100):
    print("=" * 60)
    print("🎓 THESIS EVALUATION: Baseline vs Agentic Framework")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1] Loading Test Data from: {test_csv_path}")
    if not os.path.exists(test_csv_path):
        print(f"❌ Error: Test data not found at {test_csv_path}")
        return

    # Support both .csv and .tsv
    if test_csv_path.endswith(".tsv"):
        df = pd.read_csv(test_csv_path, sep="\t")
    else:
        df = pd.read_csv(test_csv_path)

    # Standardize column names
    col_map = {}
    for c in df.columns:
        if c.lower() in ("sequence", "seq", "peptide"):
            col_map[c] = "sequence"
        if c.lower() in ("label", "class", "y"):
            col_map[c] = "label"
    df.rename(columns=col_map, inplace=True)

    if "sequence" not in df.columns or "label" not in df.columns:
        print(f"❌ Error: Could not find 'sequence' and 'label' columns. Found: {list(df.columns)}")
        return

    # Keep only valid 11-mer rows
    df = df[df["sequence"].str.len() == 11]
    df = df.sample(n=min(num_samples, len(df)), random_state=42)

    sequences = df["sequence"].tolist()
    true_labels = [int(l) for l in df["label"].tolist()]
    print(f"Loaded {len(sequences)} test samples (11-mers).")

    # 2. Initialize Framework
    print("\n[2] Initializing Agentic Framework...")
    data_dir = data_dir or _auto_data_dir()
    orch = Orchestrator(checkpoint_path=checkpoint_path, data_dir=data_dir)

    # Variables to store predictions
    baseline_probs, baseline_preds = [], []
    agentic_probs, agentic_preds = [], []

    # 3. Run Predictions
    print("\n[3] Running Predictions...")
    for seq in tqdm(sequences):
        # Analyze as a 11-mer motif
        query = f"Analyze motif: {seq}"

        result = orch.execute_analysis(query)

        # Default fallback if error
        b_prob, a_prob = 0.5, 0.5
        b_pred, a_pred = 0, 0

        if result.get("success") and result.get("all_predictions"):
            # 11-mer → exactly one R/K site at center (position 6)
            all_preds = result["all_predictions"]
            # Pick position 6 (center), else first site
            site_result = next(
                (s for s in all_preds if s.get("position") == 6),
                all_preds[0] if all_preds else None
            )

            if site_result:
                # Baseline: HybridARGME transformer probability only
                b_prob = site_result.get("P_hybrid", 0.5)
                b_pred = 1 if b_prob >= 0.5 else 0

                # Agentic: Consensus of all agents
                a_prob = site_result.get("P_final", site_result.get("probability", 0.5))
                a_pred = 1 if a_prob >= 0.5 else 0

        baseline_probs.append(b_prob)
        baseline_preds.append(b_pred)
        agentic_probs.append(a_prob)
        agentic_preds.append(a_pred)

    # 4. Calculate Metrics
    print("\n[4] Calculating Metrics...")

    def safe_auc(y_true, y_proba):
        try:
            return roc_auc_score(y_true, y_proba)
        except Exception:
            return float("nan")

    metrics = {
        "Model": ["Baseline (HybridARGME)", "Proposed (Agentic HybridARGME)"],
        "Accuracy": [
            accuracy_score(true_labels, baseline_preds),
            accuracy_score(true_labels, agentic_preds),
        ],
        "F1-Score": [
            f1_score(true_labels, baseline_preds, zero_division=0),
            f1_score(true_labels, agentic_preds, zero_division=0),
        ],
        "MCC": [
            matthews_corrcoef(true_labels, baseline_preds),
            matthews_corrcoef(true_labels, agentic_preds),
        ],
        "AUC": [
            safe_auc(true_labels, baseline_probs),
            safe_auc(true_labels, agentic_probs),
        ],
    }

    # 5. Print Output Table
    results_df = pd.DataFrame(metrics)
    print("\n" + "=" * 60)
    print("📊 FINAL THESIS RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("=" * 60)

    out_csv = "thesis_results_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to '{out_csv}'")
    return results_df


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thesis Evaluation: HybridARGME vs Agentic HybridARGME")
    parser.add_argument(
        "--test-csv", type=str, default=None,
        help="Path to test CSV/TSV with 'sequence' and 'label' columns (auto-detected if omitted)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained checkpoint (auto-detected if omitted)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Root data-11 directory for agents (auto-detected if omitted)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of test samples to evaluate (default: 100)"
    )
    args = parser.parse_args()

    test_csv  = args.test_csv  or _auto_test_csv()
    ckpt      = args.checkpoint or _auto_checkpoint()
    data_dir  = args.data_dir  or _auto_data_dir()

    if not test_csv:
        print("❌ Could not locate a test CSV. Use --test-csv to specify the path.")
        sys.exit(1)
    if not ckpt:
        print("❌ Could not locate a checkpoint. Use --checkpoint to specify the path.")
        sys.exit(1)

    evaluate_thesis(test_csv, ckpt, data_dir=data_dir, num_samples=args.num_samples)

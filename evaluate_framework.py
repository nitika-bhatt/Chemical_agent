"""
Framework Evaluation Script
===========================
Compares the performance metrics (MCC, F1, AUC, Accuracy, Precision, Recall)
of the Standalone HybridARGME model vs the Multi-Agent framework on validation/test datasets.

Usage on Colab:
  !python evaluate_framework.py --checkpoint checkpoints/pretrained_esm_v4.pt --dataset splits/combined_test.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from agents.orchestrator import Orchestrator


def evaluate_on_dataset(checkpoint_path, dataset_csv_path, output_dir, data_dir=None):
    print("=" * 80)
    print("  Evaluating Standalone HybridARGME vs Multi-Agent Framework")
    print("=" * 80)

    # 1. Load Dataset
    if not os.path.exists(dataset_csv_path):
        print(f"[ERROR] Test dataset not found at: {dataset_csv_path}")
        print("Please run train.py first to generate the dataset splits.")
        return

    df = pd.read_csv(dataset_csv_path)
    print(f"[INFO] Loaded {len(df)} candidate sites from {dataset_csv_path}")

    # Map target columns (each row represents an 11-mer window centered on the target site)
    seq_col = "sequence" if "sequence" in df.columns else "window"
    lbl_col = "label" if "label" in df.columns else ("true_label" if "true_label" in df.columns else None)

    if not lbl_col:
        print("[ERROR] CSV must contain a 'label' or 'true_label' column to evaluate accuracies.")
        return

    # 2. Initialize Orchestrator and Consensus Agent
    print("\n[INFO] Initializing Agents...")
    # Initialize the orchestrator to access components
    orch = Orchestrator(checkpoint_path=checkpoint_path, data_dir=data_dir)
    dec_agent = orch.consensus_agent
    pred_agent = orch.hybridargme_agent  # Standalone predictor

    # 3. Predict on Test Set
    print("\n[INFO] Running predictions on test dataset...")
    evaluated_sites = []

    # Group by sequence to make ESM-2 embedding calls highly efficient
    grouped = df.groupby(seq_col)

    for seq, group in tqdm(grouped, desc="Processing unique sequences"):
        # Run prediction agent on sequence
        pred_res = pred_agent.execute({"sequence": seq})
        sites = pred_res.get("predictions", [])

        if not sites:
            continue

        # Run parallel features (Motif, Structural, Conservation, Dataset Evidence)
        try:
            sites = orch._run_parallel_scoring(sites)
        except Exception as e:
            print(f"[ERROR] Scoring failed for sequence: {e}")
            continue

        # Run Consensus Fusion Agent to get final fused predictions
        fusion_res = dec_agent.execute({"sites": sites})
        final_sites = fusion_res.get("final_predictions", sites)

        # Find the predicted site at position 6 (the center target residue)
        pred_site = None
        for s in final_sites:
            if s["position"] == 6:
                pred_site = s
                break

        if pred_site:
            for _, row in group.iterrows():
                true_lbl = int(row[lbl_col])
                pred_site_copy = dict(pred_site)
                pred_site_copy["true_label"] = true_lbl
                evaluated_sites.append(pred_site_copy)

    if not evaluated_sites:
        print("[ERROR] No evaluated sites matched between dataset and predictions.")
        return

    # 4. Compute and Print Comparative Metrics Table
    print("\n" + "=" * 50)
    print("  ACCURACY COMPARISON METRICS")
    print("=" * 50)

    comparison = dec_agent.evaluate(evaluated_sites)

    if "error" in comparison:
        print(f"[ERROR] Metrics computation failed: {comparison['error']}")
        return

    metrics_df = pd.DataFrame(comparison).T
    print(metrics_df.to_string())

    # Save comparison to output folder
    os.makedirs(output_dir, exist_ok=True)
    report_csv = os.path.join(output_dir, "model_accuracy_comparison.csv")
    metrics_df.to_csv(report_csv)
    print(f"\n[SUCCESS] Saved comparative statistics table to: {report_csv}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HybridARGME vs Multi-Agent Framework")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_esm_v4.pt", help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="splits/combined_test.csv", help="Path to test CSV split")
    parser.add_argument("--data-dir", type=str, default=None, help="Root data-11 directory for agents (auto-detected if not set)")
    parser.add_argument("--output-dir", type=str, default="proteinmind_outputs", help="Output directory to save results")
    args = parser.parse_args()

    evaluate_on_dataset(args.checkpoint, args.dataset, args.output_dir, data_dir=args.data_dir)

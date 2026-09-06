"""
Explainer Faithfulness Evaluation
===================================
Turns the attention-map / SHAP-style plots your ExplainerAgent already
produces into a quantitative faithfulness metric, using a standard
deletion test:

  1. Get per-residue attribution scores for a predicted-positive site
     (from ExplainerAgent — attention_map or shap_scores).
  2. Mask the top-k highest-attributed residues in the 11-mer window
     (replace with 'X'), re-run HybridARGMEAgent, measure how much
     P_hybrid drops.
  3. Compare against masking k RANDOM residues instead (control).

If the top-attributed positions consistently cause a bigger confidence
drop than random positions, that's quantitative evidence the attributions
are actually meaningful — not just a nice-looking heatmap. This is the
kind of number a reviewer will accept in place of "the plots look
reasonable."

Usage:
    python faithfulness_eval.py --data-dir data-11 --checkpoint checkpoints/pretrained_esm_v4.pt \\
        --ptm-subdir Asymmetric_dimethylarginine --n-sites 200 --top-k 2
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.orchestrator import Orchestrator                    # noqa: E402
from cross_species_eval import SPECIES, load_species_windows   # noqa: E402


def get_attribution(exp_data: dict, window_len: int = 11) -> list:
    """
    Pulls a per-residue attribution vector from ExplainerAgent's output.
    Prefers attention_map, falls back to shap_scores. Returns None if
    neither is usable (caller should skip that site).
    """
    attr = exp_data.get("attention_map") or exp_data.get("shap_scores")
    if not attr or len(attr) != window_len:
        return None
    return [abs(float(a)) for a in attr]


def masked_confidence(agent, window: str, mask_positions: set) -> float:
    """Re-scores a window with the given 0-indexed positions replaced by 'X'."""
    masked = "".join("X" if i in mask_positions else c for i, c in enumerate(window))
    emb = agent.get_esm_embedding(masked)
    x = emb.unsqueeze(0).to(agent.device)
    probs = []
    for model in agent.models:
        with torch.no_grad():
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1)[:, 1].item())
    return float(np.mean(probs))


def original_confidence(agent, window: str) -> float:
    emb = agent.get_esm_embedding(window)
    x = emb.unsqueeze(0).to(agent.device)
    probs = []
    for model in agent.models:
        with torch.no_grad():
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1)[:, 1].item())
    return float(np.mean(probs))


def main(data_dir, checkpoint_path, ptm_subdir, n_sites, top_k, seed=0):
    random.seed(seed)
    print(f"Loading Orchestrator (checkpoint={checkpoint_path}) ...")
    orch = Orchestrator(checkpoint_path=checkpoint_path)
    agent = orch.hybridargme_agent

    # Gather positive-labeled windows (methylated) to explain — faithfulness
    # is only meaningful for sites the model actually calls positive.
    candidates = []
    for species in SPECIES:
        seqs, labels = load_species_windows(data_dir, species, ptm_subdir)
        candidates.extend([s for s, l in zip(seqs, labels) if l == 1])
    random.shuffle(candidates)

    top_k_drops, random_k_drops = [], []
    evaluated = 0

    for window in candidates:
        if evaluated >= n_sites:
            break
        if len(window) != 11:
            continue

        orig_conf = original_confidence(agent, window)
        if orig_conf < 0.5:
            continue  # only explain sites the model is confident are positive

        site_info = {"position": 6, "residue": window[5], "motif": window,
                     "P_hybrid": orig_conf, "prediction": 1, "ptm_type": "methyl"}
        exp_data = orch.explainer_agent.execute({"site_info": site_info})
        attribution = get_attribution(exp_data)
        if attribution is None:
            continue

        # Top-k most-attributed positions
        top_positions = set(
            sorted(range(len(attribution)), key=lambda i: attribution[i], reverse=True)[:top_k]
        )
        # k random positions (control), excluding the top-k set itself
        remaining = [i for i in range(len(attribution)) if i not in top_positions]
        random_positions = set(random.sample(remaining, min(top_k, len(remaining))))

        top_drop = orig_conf - masked_confidence(agent, window, top_positions)
        random_drop = orig_conf - masked_confidence(agent, window, random_positions)

        top_k_drops.append(top_drop)
        random_k_drops.append(random_drop)
        evaluated += 1

    if evaluated == 0:
        print("No sites could be evaluated — check that ExplainerAgent returns "
              "'attention_map' or 'shap_scores' of length 11, and that some "
              "windows score >=0.5 under the current checkpoint.")
        return

    df = pd.DataFrame({
        "top_k_confidence_drop":    top_k_drops,
        "random_k_confidence_drop": random_k_drops,
    })
    df["faithfulness_gap"] = df["top_k_confidence_drop"] - df["random_k_confidence_drop"]

    print(f"\nEvaluated {evaluated} confidently-positive sites, masking top-{top_k} "
          f"attributed residues vs {top_k} random residues.\n")
    print(df.describe().round(4).to_string())
    print(f"\nMean faithfulness gap (top-k drop − random-k drop): "
          f"{df['faithfulness_gap'].mean():+.4f}")
    print("A positive, meaningfully-sized gap means the explainer's top-attributed "
          "residues genuinely matter more to the prediction than random residues do — "
          "i.e. the attribution maps are faithful, not just decorative.")

    df.to_csv("faithfulness_results.csv", index=False)
    print("\nSaved: faithfulness_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data-11")
    parser.add_argument("--ptm-subdir", default="Asymmetric_dimethylarginine")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "pretrained_esm_v4.pt"))
    parser.add_argument("--n-sites", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    main(args.data_dir, args.checkpoint, args.ptm_subdir, args.n_sites, args.top_k)

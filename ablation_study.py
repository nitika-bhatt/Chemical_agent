"""
Agent Ablation Study
=====================
Quantifies how much each specialist agent (Motif, Conservation, Dataset
Evidence, Structural) actually contributes to final prediction quality,
by neutralizing one agent's score at a time and re-running consensus
fusion — using your REAL orchestrator, agents, and consensus model. This
directly produces the ablation table your thesis needs to justify the
"agentic ensemble improves accuracy" claim (the 90.73% vs 88.41% number
alone doesn't show which agents are doing the work).

──────────────────────────────────────────────────────────────────────────
ASSUMPTION THIS SCRIPT MAKES (please verify against your own agent code,
since I don't have motif_agent.py / conservation_agent.py /
dataset_evidence_agent.py / structural_agent.py to confirm this myself):

  I'm assuming MotifAgent, ConservationAgent, DatasetEvidenceAgent, and
  StructuralAgent score each site independently from its "motif" /
  "residue" string, WITHOUT depending on "position" meaning a real index
  within a specific full-length protein (StructuralAgent explicitly says
  as much in orchestrator.py's own comments — "No accession = dataset-
  centric neutral mode"). That assumption is what lets this script batch
  many independent 11-mer windows together in one _run_parallel_scoring()
  call, by giving each window a unique synthetic "position" (0, 1, 2, ...)
  purely so Orchestrator's internal index-by-position merge doesn't
  collide different windows into the same slot.

  If any of those four agents DO use "position" as a real sequence
  coordinate (e.g. for a conservation lookup keyed by protein+position),
  this batching assumption breaks and you'd need to run one window at a
  time instead (slower, but trivial to change — see `BATCH_SAFE` flag
  below).

Neutral fallback values for each ablated agent are taken directly from
orchestrator.py's own merge defaults (P_motif=0.50, P_conserv=0.25,
P_evidence=0.10, P_struct=0.50) — i.e. "this agent found no evidence
either way," not zero.

Usage:
    python ablation_study.py --data-dir data-11 --checkpoint checkpoints/pretrained_esm_v4.pt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.orchestrator import Orchestrator  # noqa: E402
from cross_species_eval import SPECIES, load_species_windows  # noqa: E402


# Set to False if your agents DO use "position" as a real protein coordinate
# (see the module docstring) — this forces one-window-at-a-time scoring
# instead of batching, which is slower but always safe.
BATCH_SAFE = True

# Orchestrator's own neutral defaults (from its site-merge code) — what each
# agent's score falls back to when we ablate it.
NEUTRAL_DEFAULTS = {
    "P_motif":    0.50,
    "P_conserv":  0.25,
    "P_evidence": 0.10,
    "P_struct":   0.50,
}

ABLATIONS = {
    "full_ensemble":      [],                # nothing ablated — baseline
    "no_motif":           ["P_motif"],
    "no_conservation":    ["P_conserv"],
    "no_dataset_evidence":["P_evidence"],
    "no_structural":      ["P_struct"],
    "hybrid_only":        ["P_motif", "P_conserv", "P_evidence", "P_struct"],
}


def build_batched_sites(orch: Orchestrator, sequences: list) -> list:
    """
    Builds one independent 'site' per window and scores them all through
    HybridARGMEAgent + the four parallel agents in ONE batched call, using
    a synthetic unique position per window (see module docstring).
    """
    agent = orch.hybridargme_agent
    embeddings = agent.get_esm_embeddings_batched(sequences)

    sites = []
    for i, (seq, emb) in enumerate(zip(sequences, embeddings)):
        x = emb.unsqueeze(0).to(agent.device)
        probs = []
        for model in agent.models:
            with torch.no_grad():
                logits = model(x)
                probs.append(torch.softmax(logits, dim=1)[:, 1].item())
        sites.append({
            "position":   i,                       # synthetic, unique per window
            "residue":    seq[5] if len(seq) > 5 else seq[len(seq) // 2],
            "motif":      seq,
            "P_hybrid":   float(np.mean(probs)),
            "prediction": int(np.mean(probs) >= agent.avg_threshold),
            "ptm_type":   "methyl",
        })

    if BATCH_SAFE:
        return orch._run_parallel_scoring(sites, accession="")
    else:
        merged = []
        for site in sites:
            merged.extend(orch._run_parallel_scoring([site], accession=""))
        return merged


def apply_ablation(sites: list, ablated_keys: list) -> list:
    """Returns a copy of sites with the given score keys forced to neutral."""
    out = []
    for s in sites:
        s2 = dict(s)
        for key in ablated_keys:
            s2[key] = NEUTRAL_DEFAULTS[key]
        out.append(s2)
    return out


def evaluate_config(orch: Orchestrator, sites: list, labels: np.ndarray) -> dict:
    result = orch.consensus_agent.execute({"sites": sites})
    final = result.get("final_predictions", sites)
    preds = np.array([s.get("label", s.get("prediction", 0)) for s in final])
    probs = np.array([s.get("P_final", s.get("P_hybrid", 0.0)) for s in final])

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, zero_division=0),
        "mcc":      matthews_corrcoef(labels, preds) if len(set(labels)) > 1 else float("nan"),
        "auc":      roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
    }


def main(data_dir, checkpoint_path, ptm_subdir, max_per_species=None):
    print(f"Loading Orchestrator (checkpoint={checkpoint_path}) ...")
    orch = Orchestrator(checkpoint_path=checkpoint_path)

    all_seqs, all_labels = [], []
    for species in SPECIES:
        seqs, labels = load_species_windows(data_dir, species, ptm_subdir)
        if max_per_species:
            seqs, labels = seqs[:max_per_species], labels[:max_per_species]
        all_seqs.extend(seqs)
        all_labels.extend(labels)

    if not all_seqs:
        print("No windows loaded — check --data-dir / --ptm-subdir.")
        return

    print(f"Loaded {len(all_seqs)} labeled windows across {len(SPECIES)} species.")
    labels = np.array(all_labels)

    print("Scoring all windows through HybridARGME + the 4 parallel agents "
          "(this happens once; ablations just re-fuse the same scores)...")
    base_sites = build_batched_sites(orch, all_seqs)

    rows = []
    for name, ablated_keys in ABLATIONS.items():
        sites = apply_ablation(base_sites, ablated_keys)
        metrics = evaluate_config(orch, sites, labels)
        metrics["configuration"] = name
        metrics["ablated"] = ", ".join(ablated_keys) if ablated_keys else "(none)"
        rows.append(metrics)
        print(f"  {name:22s}  acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  mcc={metrics['mcc']:.4f}  auc={metrics['auc']:.4f}")

    df = pd.DataFrame(rows).set_index("configuration")
    print("\n=== Ablation Results ===")
    print(df.round(4).to_string())
    df.to_csv("ablation_results.csv")
    print("\nSaved: ablation_results.csv")
    print("\nDelta vs full_ensemble (accuracy):")
    base_acc = df.loc["full_ensemble", "accuracy"]
    for name in df.index:
        if name != "full_ensemble":
            print(f"  {name:22s}  {df.loc[name, 'accuracy'] - base_acc:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data-11")
    parser.add_argument("--ptm-subdir", default="Asymmetric_dimethylarginine")
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "pretrained_esm_v4.pt"))
    parser.add_argument("--max-per-species", type=int, default=None,
                         help="Cap windows per species for a quick smoke test.")
    args = parser.parse_args()
    main(args.data_dir, args.checkpoint, args.ptm_subdir, args.max_per_species)

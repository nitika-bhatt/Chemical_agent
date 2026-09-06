r"""Print intermediate scores for one ProteinMind query.

Usage:
    .\.venv\Scripts\python.exe diagnose_single_query.py --query "Analyze motif: KTSVYRQQQPT"
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.orchestrator import Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="ProteinMind query to diagnose")
    args = parser.parse_args()

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "pretrained_esm_v4.pt")
    consensus_path = os.path.join(
        PROJECT_ROOT, "checkpoints", "consensus_agent_meta_learner.pkl"
    )
    data_dir = os.path.join(PROJECT_ROOT, "data-11")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
    print(f"Consensus model: {consensus_path}")
    print(f"Consensus model exists: {os.path.exists(consensus_path)}")

    orchestrator = Orchestrator(
        checkpoint_path=checkpoint_path,
        consensus_model_path=consensus_path,
        data_dir=data_dir,
    )
    result = orchestrator.execute_analysis(args.query)

    if not result.get("success"):
        print(f"Analysis failed: {result.get('error', 'unknown error')}")
        return 1

    summary = result["predictions_summary"]
    print(f"\nFusion mode: {summary.get('fusion_mode')}")
    print(f"Consensus threshold: {summary.get('threshold_used'):.6f}")
    print(
        "\nposition residue motif        P_hybrid  P_motif  P_evidence "
        "P_conserv  P_struct  P_final   label"
    )
    print("-" * 98)
    for site in result.get("all_predictions", []):
        print(
            f"{site.get('position', 0):>8} {site.get('residue', '?'):>7} "
            f"{site.get('motif', ''):<11} "
            f"{site.get('P_hybrid', 0.5):>8.4f}  "
            f"{site.get('P_motif', 0.5):>7.4f}  "
            f"{site.get('P_evidence', 0.5):>10.4f}  "
            f"{site.get('P_conserv', 0.5):>9.4f}  "
            f"{site.get('P_struct', 0.5):>8.4f}  "
            f"{site.get('P_final', 0.0):>7.4f}  "
            f"{site.get('label', 0):>5}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
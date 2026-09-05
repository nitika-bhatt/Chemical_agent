"""
ProteinMind CLI — Dataset-Centric Multi-Agent Methylation Analysis
==================================================================
Supports both local (Windows/Linux) and Google Colab / Drive environments.

Usage (local):
    python run_proteinmind.py --query "Analyze sequence: MKRGGKPLRTAMKRGGKPLRTA"

Usage (Colab with Drive):
    python run_proteinmind.py \\
        --query "Analyze sequence: MKRGGKPLRTA" \\
        --drive "/content/drive/MyDrive/ProteinMind_Agent" \\
        --data-dir "data-11" \\
        --checkpoint "checkpoints/pretrained_esm_v4.pt"
"""

import os
import sys
import re
import argparse
import json
from datetime import datetime

from agents.orchestrator import Orchestrator
from visuals.generator import VisualsGenerator
from report.clinical_report import ClinicalReportGenerator


def resolve_output_path(drive: str, output_dir: str) -> str:
    """
    Resolve the final output directory.
    - If drive is "." or empty, use output_dir directly (local run).
    - Otherwise join drive + output_dir (Colab/Drive run).
    """
    if drive in (".", "", None):
        path = output_dir
    else:
        path = os.path.join(drive, output_dir)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_protein_output_path(base_dir: str, query_id: str) -> str:
    """Create a stable, readable output folder for one protein analysis."""
    folder_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id or "unknown")
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="ProteinMind: Agentic PTM Methylation Analysis CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── Core query ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--query", required=True,
        help=(
            "Query for methylation analysis. Supported formats:\n"
            "  'Analyze sequence: MKRGGKPLRTA...'\n"
            "  'Analyze motif: FTLQIRGRERF'\n"
            "  'Analyze sample #125 from dataset'"
        )
    )

    # ── Optional metadata (for metadata CSV lookup) ──────────────────────────
    parser.add_argument("--metadata-dir", help="Root directory for metadata CSVs (optional)")
    parser.add_argument("--species",      help="Species name matching metadata folder (optional)")
    parser.add_argument("--ptm",          help="PTM type matching metadata folder (optional)")
    parser.add_argument("--label",        help="Label (positive/negative) for metadata file (optional)")

    # ── Model checkpoint ─────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained ensemble checkpoint (pretrained_esm_v4.pt).\n"
             "If not provided, runs with untrained weights (demo mode)."
    )

    # ── Dataset directory ─────────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir", type=str, default="data-11",
        help=(
            "Path to the dataset directory (default: 'data-11').\n"
            "On Colab with Drive: set to the full mounted path.\n"
            "Locally: relative or absolute path to the data-11 folder."
        )
    )

    # ── Output location ───────────────────────────────────────────────────────
    parser.add_argument(
        "--drive", type=str, default=".",
        help=(
            "Root Drive mount path (default: '.' = current directory).\n"
            "On Colab: set to '/content/drive/MyDrive/ProteinMind_Agent'."
        )
    )
    parser.add_argument(
        "--output-dir", type=str, default="proteinmind_outputs",
        help="Subdirectory for generated outputs (default: 'proteinmind_outputs')."
    )

    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    output_path = resolve_output_path(args.drive, args.output_dir)
    args.output_dir = output_path

    # Resolve data_dir: if drive is given and data_dir is relative, join with drive
    data_dir = args.data_dir
    if not os.path.isabs(data_dir) and args.drive not in (".", "", None):
        candidate = os.path.join(args.drive, data_dir)
        if os.path.exists(candidate):
            data_dir = candidate
    # Final check — fall back to local relative path
    if not os.path.exists(data_dir):
        local_candidate = os.path.join(os.path.dirname(__file__), data_dir)
        if os.path.exists(local_candidate):
            data_dir = local_candidate

    print("=" * 80)
    print("  ProteinMind — Agentic Protein Methylation Analytical Platform")
    print("  Dataset-Centric Multi-Agent HybridARGME Framework")
    print("=" * 80)
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Query      : {args.query}")
    print(f"  Data Dir   : {os.path.abspath(data_dir)}")
    print(f"  Output Dir : {os.path.abspath(output_path)}")
    print()

    # ── 1. Load Metadata (optional) ───────────────────────────────────────────
    metadata_dict = {
        "protein_name": "Query Protein",
        "uniprot_id":   "N/A",
        "gene_name":    "N/A",
        "organism":     "N/A",
        "function":     "Dataset-centric methylation prediction via HybridARGME."
    }

    if args.metadata_dir and args.species and args.ptm and args.label:
        try:
            import pandas as pd
            meta_path = os.path.join(
                args.metadata_dir, args.species, args.ptm,
                f"{args.species.lower()}_{args.label}_{args.ptm}_metadata.csv"
            )
            if os.path.exists(meta_path):
                df_meta = pd.read_csv(meta_path)
                if not df_meta.empty:
                    row = df_meta.iloc[0]
                    metadata_dict = {
                        "protein_name": str(row.get("Protein Name", "N/A")),
                        "uniprot_id":   str(row.get("UniProt ID",   "N/A")),
                        "gene_name":    str(row.get("Gene Symbol",  "N/A")),
                        "organism":     str(row.get("Organism",     "N/A")) if "Organism" in df_meta.columns else "N/A",
                        "function":     str(row.get("Function",     "N/A")) if "Function" in df_meta.columns else "N/A",
                    }
                    print(f"[CLI] Loaded metadata from: {meta_path}")
            else:
                print(f"[CLI WARNING] Metadata file not found: {meta_path}")
        except Exception as e:
            print(f"[CLI WARNING] Failed to load metadata: {e}")

    uniprot_id = metadata_dict.get("uniprot_id", "").strip()
    if uniprot_id in ("N/A", ""):
        uniprot_id = ""

    # ── 2. Run Orchestrator ───────────────────────────────────────────────────
    try:
        orch = Orchestrator(checkpoint_path=args.checkpoint, data_dir=data_dir)
        analysis_res = orch.execute_analysis(args.query, accession=uniprot_id)
    except Exception as e:
        import traceback
        print(f"\n[CLI FATAL] Failed to initialize/run Orchestrator:")
        traceback.print_exc()
        sys.exit(1)

    if not analysis_res.get("success", False):
        print(f"\n[CLI ERROR] Pipeline failed: {analysis_res.get('error')}")
        sys.exit(1)

    # Attach metadata to payload
    analysis_res["metadata"] = metadata_dict
    # Preserve the accession resolved by the orchestrator for direct accession queries.
    if uniprot_id:
        analysis_res["uniprot_id"] = uniprot_id

    # ── 3. Print Summary ──────────────────────────────────────────────────────
    query_id   = analysis_res.get("query_id",   "unknown")
    query_mode = analysis_res.get("query_mode", "unknown")
    ptm_type   = analysis_res.get("ptm_type",   "methyl")
    summary    = analysis_res.get("predictions_summary", {})
    args.output_dir = resolve_protein_output_path(args.output_dir, query_id)

    print("\n" + "-" * 60)
    print("  ANALYSIS RESULTS SUMMARY")
    print("-" * 60)
    print(f"  Query ID      : {query_id}")
    print(f"  Query Mode    : {query_mode}")
    print(f"  PTM Scored    : {ptm_type.upper()}")
    print(f"  Sites Checked : {summary.get('total_sites_checked', 'N/A')}")
    print(f"  PTM-Positive  : {summary.get('positive_sites_count', 0)} sites")
    thr = summary.get('threshold_used', 0.5)
    print(f"  Threshold     : {thr:.3f}" if isinstance(thr, float) else f"  Threshold     : {thr}")
    print(f"  Fusion Mode   : {summary.get('fusion_mode', 'N/A')}")

    positives = analysis_res.get("positive_predictions", [])
    if positives:
        print("\n  Predicted Methylation Sites:")
        for p in positives:
            prob = p.get("P_final", p.get("probability", float("nan")))
            print(f"    ✓ Position {p['position']:4d} | {p['residue']} | {p['motif']} | Score: {prob:.4f} [{p.get('confidence','N/A')}]")
    else:
        print("\n  No sites exceeded the prediction confidence threshold.")

    # ── 4. Generate Visualizations ────────────────────────────────────────────
    print(f"\n[CLI] Generating visualizations → '{args.output_dir}'...")
    generator    = VisualsGenerator(args.output_dir)
    visual_files = generator.execute_all(analysis_res)

    print("\n  Visuals Generated:")
    for key, val in visual_files.items():
        if isinstance(val, list):
            for i, p in enumerate(val):
                if p:
                    print(f"    [{key}][{i+1}] : {p}")
        elif val:
            print(f"    [{key}] : {val}")

    # ── 5. Generate PDF Report ────────────────────────────────────────────────
    pdf_path = os.path.join(args.output_dir, f"report_{query_id}.pdf")
    print(f"\n[CLI] Compiling PDF report → '{pdf_path}'...")
    try:
        rep_gen = ClinicalReportGenerator(pdf_path)
        rep_gen.generate_pdf(analysis_res, visual_files)
    except Exception as e:
        import traceback
        print(f"\n[CLI ERROR] Failed to write PDF report:")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  ✅ ProteinMind Analysis Complete!")
    print(f"  📄 PDF Report : {os.path.abspath(pdf_path)}")
    print(f"  📁 All Files  : {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# ============================================================
#  ProteinMind — Google Colab Setup & Run Script
#  Copy-paste this into a Colab notebook cell-by-cell
# ============================================================

# ─── CELL 1: Mount Google Drive ──────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─── CELL 2: Install dependencies ────────────────────────────
# Run once per Colab session
import subprocess
subprocess.run([
    'pip', 'install', '-q',
    'torch>=2.0.0',
    'fair-esm>=2.0.0',
    'reportlab>=4.0.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'scikit-learn>=1.2.0',
    'pandas>=1.5.0',
    'numpy>=1.22.0',
    'joblib>=1.3.0',
    'requests>=2.30.0',
], check=True)
print("All dependencies installed!")

# ─── CELL 3: Set project path ────────────────────────────────
import sys, os

# EDIT THIS: path where you uploaded ProteinMind_Agent on Drive
DRIVE_ROOT   = "/content/drive/MyDrive/ProteinMind_Agent"
DATA_DIR     = os.path.join(DRIVE_ROOT, "data-11")
OUTPUT_DIR   = os.path.join(DRIVE_ROOT, "proteinmind_outputs")
CHECKPOINT   = os.path.join(DRIVE_ROOT, "checkpoints", "pretrained_esm_v4.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add project to Python path
sys.path.insert(0, DRIVE_ROOT)
os.chdir(DRIVE_ROOT)

print(f"Project root : {DRIVE_ROOT}")
print(f"Data dir     : {DATA_DIR}  (exists: {os.path.exists(DATA_DIR)})")
print(f"Checkpoint   : {CHECKPOINT}  (exists: {os.path.exists(CHECKPOINT)})")
print(f"Output dir   : {OUTPUT_DIR}")

# ─── CELL 4: Run full analysis ────────────────────────────────
# !! EDIT THE QUERY BELOW !!
QUERY = "Analyze sequence: MKRGGKPLRTAMKRGGKPLRTASRGGRGGFGSRGGRGG"

# ------ Run pipeline ------
from agents.orchestrator import Orchestrator
from visuals.generator import VisualsGenerator
from report.clinical_report import ClinicalReportGenerator

import hashlib, os

print("\n" + "="*70)
print("  ProteinMind — Running Analysis")
print("="*70)

orch = Orchestrator(
    checkpoint_path=CHECKPOINT if os.path.exists(CHECKPOINT) else None,
    data_dir=DATA_DIR
)

result = orch.execute_analysis(QUERY)

if not result.get("success"):
    print(f"\n[ERROR] {result.get('error')}")
else:
    # Add metadata
    result["metadata"] = {
        "protein_name": "Query Protein (Novel Sequence)",
        "gene_name":    "N/A",
        "organism":     "Homo sapiens",
        "function":     "Dataset-centric methylation prediction via HybridARGME.",
    }
    result["uniprot_id"] = "N/A"

    print(f"\nPositive sites found: {result['predictions_summary']['positive_sites_count']}")
    for p in result.get("positive_predictions", []):
        print(f"  {p['residue']}{p['position']}  P_final={p.get('P_final',0):.4f}  [{p.get('confidence','N/A')}]")

    # Generate visuals
    gen     = VisualsGenerator(output_dir=OUTPUT_DIR)
    visuals = gen.execute_all(result)

    # Generate PDF
    qid      = result.get("query_id", "query")
    pdf_path = os.path.join(OUTPUT_DIR, f"report_{qid}.pdf")
    rep      = ClinicalReportGenerator(pdf_path)
    rep.generate_pdf(result, visuals)

    print(f"\nPDF saved: {pdf_path}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Download PDF directly in Colab
    from google.colab import files
    files.download(pdf_path)

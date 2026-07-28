# ProteinMind: Multi-Agent Interpretability & Reporting Platform for PTM Site Prediction

ProteinMind is an agent-based bioinformatics framework built on top of `DeepTESite` (a custom deep transformer network featuring Bi-directional Multi-head Attention Mechanism (BMAM) and Adaptive Position Encoding (AASSPE)). Using precomputed **ESM-2** representations, the system predicts post-translational modifications (specifically Lysine/Arginine methylation) and employs an agentic pipeline to retrieve biological context, generate structural interpretability models, and write clinical reports automatically.

This project forms a complete **MTech Thesis workflow** designed for high-impact publication in computational biology.

---

## 1. Project Architecture

The system uses a **ReAct (Reasoning and Acting)** orchestration loop dividing operations among specialized modules:

```
d:/mtechthesis/New folder/
├── model_core.py          # Core model architecture (DeepTESite, BMAM, AASSPE, FocalLoss)
├── train.py                # Stratified 5-Fold SWA CV training script with ESM-2 caching
├── run_proteinmind.py     # Single-command CLI query-to-report pipeline execution
├── requirements.txt        # Package dependencies
├── README.md               # Project documentation & publication guide
├── SETUP_COLAB.ipynb       # Step-by-step Colab deployment notebook
├── agents/
│   ├── retrieval_agent.py  # UniProt annotation parser & PubMed E-utilities searcher
│   ├── prediction_agent.py # Sliced-window candidate site prediction engine
│   └── explainer_agent.py  # BMAM attention extractor & SHAP-like occlusion importance
├── visuals/
│   └── generator.py        # Heatmap, attribution bars, disease radar, & 3D HTML structure
├── report/
│   └── clinical_report.py  # Custom ReportLab PDF cover-page reporting compiler
└── api/
    └── app.py              # FastAPI server serving analysis, download, & 3D viewer
```

---

## 2. Quick Setup & Installation

### Local / Server Installation
Ensure Python 3.8+ is installed. Clone this repository and install packages:

```bash
pip install -r requirements.txt
```

### Dataset Structure (For Training)
Create a directory structure in the root under `data-11/`:
```
data-11/
├── Human/
│   └── Asymmetric_dimethylarginine/
│       ├── positive.csv
│       └── negative.csv
├── Chimpanzee/
│   └── ...
├── Mouse/
│   └── ...
└── Rat/
    └── ...
```
Each CSV should contain a column titled `sequence` containing 11-mer protein sequences.

---

## 3. Execution Workflows

### Workflow A: Model Training & Evaluation
To run Stratified 5-Fold Cross Validation with SWA on your datasets:
```bash
python train.py
```
This script:
1. Splits raw CSVs into balanced trainval/test datasets per species.
2. Downloads ESM-2 weights from Torch Hub and precomputes embeddings (stored in `esm_cache/`).
3. Trains the 5-fold ensemble, saving parameters and thresholds to `checkpoints/pretrained_esm_v4.pt`.
4. Outputs benchmark performance reports to `checkpoints/test_results_esm_ensemble.csv`.

### Workflow B: CLI Query Analysis (Single-Command Run)
Once you have the trained weights in `checkpoints/`, run an agentic analysis:
```bash
python run_proteinmind.py --query "Analyze P04637 for methylation"
```
This runs the Orchestrator which:
1. Queries UniProt to extract p53 (P04637) sequence, name, and functions.
2. Queries PubMed to find literature correlations.
3. Scores all Lysine/Arginine residues using the ESM-2 ensemble.
4. Explains the positive sites (Attention weights + SHAP occlusion).
5. Generates the 3D Molecular viewer, heatmap plots, and compiles `clinical_report_P04637.pdf` in the `proteinmind_outputs/` folder.

### Workflow C: FastAPI Backend Portal
To start the REST API server:
```bash
python api/app.py
```
Or use uvicorn:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
You can access Swagger API documentation at `http://localhost:8000/docs`.
- Submit POST `/api/analyze` request body: `{"query": "Analyze P04637"}`.
- Open `http://localhost:8000/api/viewer/P04637` to render the interactive 3D protein structure.

---

## 4. Thesis & Publication Publication Guidelines
When writing your thesis or manuscript, focus on these main highlights:
1. **Explainable Agentic Workflow:** Contrast this with static classifiers. Emphasize that the system automates a human biologist's manual annotation pipeline.
2. **ESM-2 Embeddings vs. Hand-Crafted Motifs:** Explain how leveraging structural/evolutionary embeddings from ESM-2 yields robust cross-species generalization.
3. **BMAM Bidirectional Attention Matrix:** Highlight how BMAM captures non-local interactions between amino acids, backed by the visual heatmaps.
4. **Residue Attribution Validation:** Correlate the SHAP occlusion scores with active binding pockets (evident in the 3D viewer) to prove model biological correctness.

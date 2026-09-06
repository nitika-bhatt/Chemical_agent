# ArgMe-XAI Agent: Arginine Methylation Explainable AI Agent

The **ArgMe-XAI Agent** is an explainable AI system for identifying possible arginine and lysine methylation sites in protein sequences. **PLMArgMe** is used as the base model for protein-language representation and candidate methylation-site prediction. The system combines this prediction with motif analysis, conservation, dataset similarity, structural information, and consensus scoring to produce an interpretable result.

The project combines these components into an explainable AI workflow with prediction, visual explanation, interactive 3D structure, and research report generation for M.Tech thesis evaluation.

---

## 1. Project Architecture

The system is organized as a **ReAct (Reasoning and Acting)** orchestration loop that divides the analysis among specialized modules:

```
ProteinMind_Agent/
├── model_core.py          # PLMArgMe model architecture and training components
├── train.py                # Stratified 5-Fold SWA CV training script with ESM-2 caching
├── run_proteinmind.py     # Single-command CLI query-to-report pipeline execution
├── requirements.txt        # Package dependencies
├── README.md               # Project documentation & publication guide
├── SETUP_COLAB.ipynb       # Step-by-step Colab deployment notebook
├── agents/
│   ├── retrieval_agent.py  # UniProt annotation parser & PubMed E-utilities searcher
│   ├── prediction_agent.py # Sliced-window candidate site prediction engine
│   └── explainer_agent.py  # Attention and SHAP-like occlusion explanations
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
The project uses Python 3.10+ and installs its dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Dataset Structure (For Training)
The dataset is stored under `data-11/`. It contains species-specific folders and positive/negative sequence CSV files for 11-residue methylation windows:
```
data-11/
├── Human/
│   └── Asymmetric_dimethylarginine/
│       ├── human_positive_*_sequences.csv
│       └── human_negative_*_sequences.csv
├── Chimpanzee/
│   └── ...
├── Mouse/
│   └── ...
└── Rat/
    └── ...
```
The sequence columns are named `Positive_sequence` or `Negative_sequence`. Each sequence window is centered on a candidate methylation residue.

---

## 3. Execution Workflows

### Workflow A: Model Training and Evaluation
Model training uses Stratified 5-Fold Cross Validation with SWA. The command is:
```bash
python train.py
```
This training workflow:
1. Splits raw CSVs into balanced trainval/test datasets per species.
2. Downloads ESM-2 weights from Torch Hub and precomputes embeddings (stored in `esm_cache/`).
3. Trains the 5-fold ensemble, saving parameters and thresholds to `checkpoints/pretrained_esm_v4.pt`.
4. Outputs benchmark performance reports to `checkpoints/test_results_esm_ensemble.csv`.

### Workflow B: CLI Query Analysis
After the dependencies and checkpoint are prepared, an analysis can be run with:
```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe run_proteinmind.py --query "Analyze P04637 for methylation"
```
The analysis workflow:
1. Resolves the complete P04637 sequence from UniProt.
2. Scores arginine and lysine candidate sites using PLMArgMe and the evidence agents.
3. Explains the highest-ranked sites using attention and occlusion attribution.
4. Retrieves an AlphaFold structure when available and highlights predicted methylation sites.
5. Saves the PDF report, interactive viewer, and plots in `proteinmind_outputs/P04637/`.

For an 11-residue dataset window, use the motif format:
```powershell
.\.venv\Scripts\python.exe run_proteinmind.py --query "Analyze motif: FTLQIRGRERF"
```
The motif output is saved in a folder such as `proteinmind_outputs/motif_FTLQIR/`.

### Workflow C: FastAPI Backend Portal
The REST API server can be started with:
```bash
.\.venv\Scripts\python.exe api/app.py
```
Or use uvicorn:
```bash
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
You can access Swagger API documentation at `http://localhost:8000/docs`.
- Submit POST `/api/analyze` request body: `{"query": "Analyze P04637"}`.
- Open `http://localhost:8000/api/viewer/P04637` to render the interactive 3D protein structure.

### Workflow D: Streamlit Web Interface

The web application can be started with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_app.py --server.port 8502
```

Open `http://localhost:8502` and enter a UniProt accession such as `P04637` for a real AlphaFold structure, or an 11-residue motif such as `FTLQIRGRERF` for a synthetic peptide visualization.

Each Streamlit analysis is saved in its own folder under `proteinmind_outputs/`.

### Workflow E: Render Deployment

The FastAPI service can start without the large model and dataset files, but
analysis requires both of these assets at runtime:

```text
checkpoints/pretrained_esm_v4.pt
checkpoints/consensus_agent_meta_learner.pkl
data-11/
```

These files are excluded from Git because of their size. Configure Render to
mount or download them during deployment, then set the service start command:

```bash
uvicorn api.app:app --host 0.0.0.0 --port $PORT
```

The root endpoint reports `analysis_ready: false` and `/api/analyze` returns
HTTP 503 when the required assets are unavailable. This prevents deployment
from silently running inference with untrained model weights.

---

## 4. Thesis Work Completed and Planned

### Implemented

1. PLMArgMe-based methylation prediction pipeline.
2. Motif, conservation, dataset-evidence, and structural analysis agents.
3. Consensus scoring to combine evidence from the agents.
4. Attention and SHAP-like occlusion explanations for predicted sites.
5. Static plots, interactive 3D visualization, and PDF report generation.
6. Protein-specific output folders under `proteinmind_outputs/`.

### Planned thesis work

1. Evaluate the model using the final training and held-out test splits.
2. Compare PLMArgMe with suitable baseline methods.
3. Analyze cross-species generalization across Human, Mouse, Rat, and Chimpanzee data.
4. Validate the highest-confidence predictions against published or experimental annotations where available.
5. Document the final evaluation metrics, limitations, and reproducibility results in the thesis.

## 5. Accuracy and Expected Contribution

The existing model comparison output evaluates the PLMArgMe-only prediction against the proposed ArgMe-XAI Agent. On the evaluated comparison set, the results were:

| System | Accuracy | F1-score | MCC | AUC |
|---|---:|---:|---:|---:|
| PLMArgMe-only baseline | 88.41% | 0.8866 | 0.7690 | 0.9255 |
| ArgMe-XAI Agent | **90.73%** | **0.9033** | **0.8174** | **0.9645** |

The ArgMe-XAI Agent therefore improved accuracy by **2.32 percentage points** on this comparison. It also improved F1-score, Matthews correlation coefficient, and area under the ROC curve. These additional metrics are important because methylation datasets may contain class imbalance, meaning that accuracy alone can give an incomplete picture of performance.

### Why the agentic system may perform better

The PLMArgMe model provides the main sequence-based prediction. The ArgMe-XAI Agent adds complementary evidence before making the final decision:

1. **Motif evidence:** identifies local sequence patterns associated with methylation.
2. **Conservation evidence:** checks whether similar sites occur across the available species.
3. **Dataset evidence:** compares the input window with known methylated and non-methylated examples.
4. **Structural evidence:** considers residue accessibility when a full protein structure is available.
5. **Consensus scoring:** combines the independent evidence sources rather than relying on one score.

This design can improve robustness because a prediction supported by several independent signals is less dependent on one local sequence pattern. It also makes the prediction easier to interpret and investigate biologically.

The reported improvement is an evaluation result, not a guarantee that every protein will be predicted more accurately. For the final thesis claim, the comparison should be rerun on identical held-out data, with confidence intervals reported where possible, together with accuracy, F1-score, MCC, sensitivity, specificity, and AUC.

## 6. Understanding the Outputs

Every analysis creates a protein-specific folder containing:

- `report_<query_id>.pdf`: Main research report with prediction tables, figures, explanations, and limitations.
- `structure_viewer_<query_id>.html`: Interactive 3D viewer with Spectrum, PTM Sites, Surface, Stick, and Reset controls.
- `structure_3d.png`: Static 3D backbone visualization.
- `structure_2d.png`: Linear sequence/structure overview.
- `sequence_annotation.png`: Candidate-site map and final scores.
- `agent_score_heatmap.png`: Evidence scores from the analysis agents.
- `attention_site_*.png`: Attention-based local residue explanations.
- `shap_site_*.png`: Occlusion-based residue attribution plots.
- `radar_site_*.png` and `site_card_*.png`: Per-site evidence summaries.

In my visualizations, predicted methylation sites are displayed as red markers. These are computational predictions and require experimental validation.

For a short 11-residue motif, the 3D structure is a synthetic peptide-like visualization. It shows the sequence and predicted site positions but should not be interpreted as an experimentally resolved protein fold.

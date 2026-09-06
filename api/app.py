import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import Orchestrator
from visuals.generator import VisualsGenerator
from report.clinical_report import ClinicalReportGenerator

# ==========================================
# FastAPI Initialization
# ==========================================
app = FastAPI(
    title="ProteinMind API Portal",
    description="Thesis Project Platform: Agentic PTM predictions, biological database retrieval, explanations, and reporting.",
    version="1.0.0"
)

# CORS support for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Caching Orchestrator on Startup
# ==========================================
# Locate default checkpoint
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_path = os.path.join(base_dir, "checkpoints", "pretrained_esm_v4.pt")
data_dir = os.path.join(base_dir, "data-11")

orchestrator = None
startup_error = None


def _deployment_error():
    missing = []
    if not os.path.isfile(checkpoint_path):
        missing.append("checkpoints/pretrained_esm_v4.pt")
    if not os.path.isdir(data_dir):
        missing.append("data-11/")
    if missing:
        return (
            "Required model/data assets are missing from the deployment: "
            + ", ".join(missing)
            + ". Upload them through Render persistent storage or configure "
              "an artifact download during the build; do not run inference "
              "with untrained weights."
        )
    return None

@app.on_event("startup")
def startup_event():
    global orchestrator, startup_error
    print("[API] Starting up. Initializing model orchestrator...")
    startup_error = _deployment_error()
    if startup_error:
        print(f"[API] Configuration error: {startup_error}")
        return
    try:
        orchestrator = Orchestrator(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
        )
    except Exception as exc:
        startup_error = f"Orchestrator initialization failed: {exc}"
        print(f"[API] {startup_error}")
        return
    print("[API] Orchestrator successfully initialized.")

# ==========================================
# Request/Response Schemas
# ==========================================
class QueryRequest(BaseModel):
    query: str  # Accepted formats:
                # • "Analyze motif: FTLQIRGRERF"
                # • "Analyze sequence: MKRGGKPLRTA..."
                # • "Analyze sample #125 from dataset"
                # • Bare amino acid sequence string

# ==========================================
# Routes
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "Online" if orchestrator is not None else "Degraded",
        "analysis_ready": orchestrator is not None,
        "startup_error": startup_error,
        "system": "ProteinMind Agentic Framework (Dataset-Centric)",
        "version": "3.0",
        "query_formats": [
            "Analyze motif: FTLQIRGRERF",
            "Analyze sequence: MKRGGKPLRTA...",
            "Analyze sample #125 from dataset",
            "Bare amino acid sequence string",
        ],
        "endpoints": {
            "/api/analyze": "POST - Runs prediction, evidence search, consensus fusion, and explainability pipeline",
            "/api/report/{query_id}": "GET - Downloads generated analysis PDF report",
            "/api/viewer/{query_id}": "GET - Serves interactive 3D structure viewer HTML"
        }
    }

@app.post("/api/analyze")
def analyze_protein(payload: QueryRequest):
    """Triggers the full dataset-centric multi-agent prediction and explanation pipeline."""
    global orchestrator
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail=startup_error or "Orchestrator model is still loading.",
        )

    query = payload.query
    try:
        # 1. Run orchestrator analysis
        analysis_res = orchestrator.execute_analysis(query)
        if not analysis_res.get("success", False):
            raise HTTPException(status_code=400,
                                detail=analysis_res.get("error", "Unknown pipeline error"))

        query_id = analysis_res["query_id"]

        # 2. Generate visuals
        plot_dir   = os.path.join(base_dir, "output_plots")
        generator  = VisualsGenerator(plot_dir)
        visual_files = generator.execute_all(analysis_res)

        # 3. Compile PDF Report
        report_dir = os.path.join(base_dir, "output_reports")
        os.makedirs(report_dir, exist_ok=True)
        pdf_path   = os.path.join(report_dir, f"report_{query_id}.pdf")

        rep_gen = ClinicalReportGenerator(pdf_path)
        rep_gen.generate_pdf(analysis_res, visual_files)

        # Add relative visual URLs for frontend retrieval
        analysis_res["visuals"] = {
            "pdf_report_url": f"/api/report/{query_id}",
            "3d_viewer_url":  f"/api/viewer/{query_id}",
        }

        return analysis_res
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.get("/api/report/{query_id}")
def get_pdf_report(query_id: str):
    """Downloads the compiled PDF report for the given query ID."""
    pdf_path = os.path.join(base_dir, "output_reports", f"report_{query_id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"No PDF report found for query ID '{query_id}'. Run analysis first."
        )
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"ProteinMind_Report_{query_id}.pdf"
    )

@app.get("/api/viewer/{query_id}", response_class=HTMLResponse)
def get_3d_viewer(query_id: str):
    """Serves the interactive 3D HTML molecular structure viewer (if available)."""
    html_path = os.path.join(base_dir, "output_plots", f"structure_viewer_{query_id}.html")
    if not os.path.exists(html_path):
        try:
            generator = VisualsGenerator(os.path.join(base_dir, "output_plots"))
            generator.generate_3d_molecular_html(query_id, [])
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Failed to generate 3D viewer: {e}")

    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="3D viewer file not found.")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

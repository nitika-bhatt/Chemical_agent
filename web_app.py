import os
import sys
import tempfile
import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Add the parent directory to Python path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.orchestrator import Orchestrator
    from visuals.generator import VisualsGenerator
    from report.clinical_report import ClinicalReportGenerator
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure you are running this from the ProteinMind_Agent directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="ProteinMind Agentic Framework", page_icon="🧬", layout="wide")

st.title("🧬 ProteinMind: Agentic PTM Analytical Platform")
st.markdown("""
Welcome to the Web Interface of the ProteinMind Framework. 
Enter a protein sequence below, and the multi-agent system will analyze it for Arginine Methylation sites using a dataset-centric evidence approach.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
checkpoint_path_input = st.sidebar.text_input(
    "Checkpoint Path", 
    value="/content/drive/MyDrive/ProteinMind_Agent/checkpoints/pretrained_esm_v4.pt"
)

# --- Main Interface ---
st.subheader("Protein Input")
sequence_input = st.text_area(
    "Enter raw Amino Acid Sequence:", 
    height=150, 
    placeholder="Example: MKRGGKPLRTARGGRVVTS..."
)

if st.button("🚀 Analyze Protein", use_container_width=True):
    if not sequence_input.strip():
        st.warning("Please enter a protein sequence first!")
    else:
        with st.spinner("Initializing Agentic Framework & Loading Data..."):
            try:
                orch = Orchestrator(checkpoint_path=checkpoint_path_input)
            except Exception as e:
                st.error(f"Failed to load Orchestrator or Checkpoint: {e}")
                st.stop()
                
        with st.spinner("Agents are analyzing the sequence..."):
            query = f"Analyze sequence: {sequence_input.strip()}"
            analysis_res = orch.execute_analysis(query)
            
            if not analysis_res.get("success", False):
                st.error(f"Analysis Failed: {analysis_res.get('error')}")
                st.stop()
                
        st.success("Analysis Complete!")
        
        # --- Results Summary ---
        st.header("📊 Analysis Results Summary")
        summary = analysis_res["predictions_summary"]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PTM Type", analysis_res["ptm_type"].upper())
        col2.metric("Sites Checked", summary.get("total_sites_checked", 0))
        col3.metric("Positive Sites Found", summary.get("positive_sites_count", 0))
        col4.metric("Threshold", f"{summary.get('threshold_used', 0.5):.3f}")
        
        # --- Positive Predictions Table ---
        positives = analysis_res.get("positive_predictions", [])
        if positives:
            st.subheader("✅ Identified Methylation Sites")
            df = pd.DataFrame(positives)
            # Reformat df for display
            display_df = df[["position", "residue", "motif"]].copy()
            display_df["P_Final Score"] = [p.get("probability", p.get("P_final", 0)) for p in positives]
            st.dataframe(display_df.style.highlight_max(axis=0, subset=["P_Final Score"]), use_container_width=True)
        else:
            st.info("No methylation sites were identified above the confidence threshold.")
            
        # --- Generate Visuals & Report ---
        with st.spinner("Generating 3D Structures and PDF Report..."):
            output_dir = tempfile.mkdtemp()
            generator = VisualsGenerator(output_dir)
            visual_files = generator.execute_all(analysis_res)
            
            pdf_path = os.path.join(output_dir, f"report_{analysis_res.get('query_id', 'output')}.pdf")
            rep_gen = ClinicalReportGenerator(pdf_path)
            rep_gen.generate_pdf(analysis_res, visual_files)
            
        # --- Display 3D Viewer ---
        html_file = visual_files.get("3d_viewer")
        if html_file and os.path.exists(html_file):
            st.header("🧬 3D Interactive Protein Structure")
            st.caption("Using ESMFold API. Predicted sites are highlighted in red spheres.")
            with open(html_file, 'r', encoding='utf-8') as f:
                html_code = f.read()
            components.html(html_code, height=650, scrolling=True)
            
        # --- Display Images ---
        st.header("📈 Agentic Explanations & Plots")
        colA, colB = st.columns(2)
        
        with colA:
            hm_img = visual_files.get("agent_score_heatmap")
            if hm_img and os.path.exists(hm_img):
                st.image(hm_img, caption="Multi-Agent Score Heatmap", use_column_width=True)
        
        with colB:
            seq_plot = visual_files.get("sequence_annotation")
            if seq_plot and os.path.exists(seq_plot):
                st.image(seq_plot, caption="Sequence Linear Annotation", use_column_width=True)
                
        # Radar charts
        radar_charts = visual_files.get("radar_charts", [])
        if radar_charts:
            st.subheader("Radar Charts for Top Sites")
            cols = st.columns(min(3, len(radar_charts)))
            for i, r_path in enumerate(radar_charts[:3]):
                with cols[i]:
                    st.image(r_path, use_column_width=True)
        
        # --- PDF Download Button ---
        st.header("📑 Download Final Clinical Report")
        with open(pdf_path, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            
        st.download_button(
            label="📄 Download PDF Report",
            data=PDFbyte,
            file_name="ProteinMind_Report.pdf",
            mime="application/octet-stream",
            help="Download the complete MTech Thesis Clinical Report"
        )

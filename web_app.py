import os

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="ArgMe-XAI Agent",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 ArgMe-XAI Agent")
st.caption("Explainable arginine and lysine methylation analysis")
st.info(f"Backend: {BACKEND_URL}")

sequence_input = st.text_area(
    "Protein sequence or UniProt accession",
    height=150,
    placeholder="Examples: P04637, FTLQIRGRERF, or PLKRLRLGPPD",
)

if st.button("🚀 Analyze protein", width="stretch"):
    query = sequence_input.strip()
    if not query:
        st.warning("Enter a protein sequence or UniProt accession first.")
        st.stop()

    if len(query.split()) == 1 and len(query) <= 15:
        if len(query) == 11:
            api_query = f"Analyze motif: {query}"
        else:
            api_query = f"Analyze {query} for methylation"
    else:
        api_query = f"Analyze sequence: {query.replace(' ', '')}"

    try:
        with st.spinner("Running analysis on the backend..."):
            response = requests.post(
                f"{BACKEND_URL}/api/analyze",
                json={"query": api_query},
                timeout=300,
            )
        response.raise_for_status()
        analysis = response.json()
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        st.code(
            "Confirm that the backend is online and that BACKEND_URL is set to "
            "https://chemical-agent.onrender.com."
        )
        st.stop()

    st.success("Analysis complete")
    summary = analysis.get("predictions_summary", {})
    columns = st.columns(4)
    columns[0].metric("PTM type", analysis.get("ptm_type", "methyl").upper())
    columns[1].metric("Sites checked", summary.get("total_sites_checked", 0))
    columns[2].metric("Positive sites", summary.get("positive_sites_count", 0))
    columns[3].metric("Threshold", f"{summary.get('threshold_used', 0.5):.3f}")

    positives = analysis.get("positive_predictions", [])
    if positives:
        st.subheader("Identified methylation sites")
        display_columns = ["position", "residue", "motif", "P_hybrid", "P_final", "confidence"]
        available_columns = [column for column in display_columns if column in positives[0]]
        st.dataframe(pd.DataFrame(positives)[available_columns], width="stretch")
    else:
        st.info("No methylation sites were identified above the confidence threshold.")

    visuals = analysis.get("visuals", {})
    pdf_url = visuals.get("pdf_report_url")
    viewer_url = visuals.get("3d_viewer_url")

    if viewer_url:
        st.subheader("Interactive structure viewer")
        try:
            viewer_response = requests.get(
                f"{BACKEND_URL}{viewer_url}", timeout=60
            )
            viewer_response.raise_for_status()
            st.html(viewer_response.text, unsafe_allow_javascript=True)
        except requests.RequestException as exc:
            st.warning(f"Structure viewer unavailable: {exc}")

    if pdf_url:
        try:
            pdf_response = requests.get(f"{BACKEND_URL}{pdf_url}", timeout=60)
            pdf_response.raise_for_status()
            st.download_button(
                "📄 Download PDF report",
                data=pdf_response.content,
                file_name=f"ArgMe-XAI_{analysis.get('query_id', 'report')}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        except requests.RequestException as exc:
            st.warning(f"PDF report unavailable: {exc}")

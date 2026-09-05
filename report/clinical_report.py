"""
Clinical Report Generator — ArgMe-XAI Agent
==========================================
Builds a complete multi-page PDF report for the ArgMe-XAI Agent methylation
prediction analysis.

FIXED in v4:
  • ALL figure blocks now placed BEFORE doc.build() — they were dead code before
  • Protein name / gene / organism shown correctly (not hash fallback)
  • Full input sequence displayed in PDF
  • 2D structure PNG embedded in PDF (new Section 4)
  • Site detail cards embedded per positive site (new Section 5)
  • Agent score heatmap, sequence annotation, radar charts all included
  • Proper section ordering and page layout
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# ─────────────────────────────────────────────────────────────────────────────
# Numbered canvas for header / footer on every page
# ─────────────────────────────────────────────────────────────────────────────

class NumberedCanvas(canvas.Canvas):
    """Computes total page count dynamically for footer page numbering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_chrome(num_pages)
            super().showPage()
        super().save()

    def _draw_page_chrome(self, page_count):
        self.saveState()

        # ── Header ────────────────────────────────────────────────────────────
        self.setStrokeColor(colors.HexColor("#1e3a8a"))
        self.setLineWidth(1.0)
        self.line(54, 738, 558, 738)

        self.setFillColor(colors.HexColor("#1e3a8a"))
        self.setFont("Helvetica-Bold", 9)
        self.drawString(54, 742, "ProteinMind  ·  Agentic PTM Analytical Platform")

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(558, 742, "ArgMe-XAI Agent  |  M.Tech Thesis")

        # ── Footer ────────────────────────────────────────────────────────────
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 38, f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        self.drawRightString(558, 38,
            f"Page {self._pageNumber} of {page_count}  ·  CONFIDENTIAL — Research Use Only")

        self.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — safely add an image only if the file exists and is non-empty
# ─────────────────────────────────────────────────────────────────────────────

def _safe_image(path: str, width: float, height: float):
    """Return an Image flowable if path is valid, else return None."""
    if path and os.path.exists(path) and os.path.getsize(path) > 0:
        return Image(path, width=width, height=height)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Report Generator
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalReportGenerator:
    """Assembles a complete multi-section PDF for one ProteinMind analysis."""

    # ── Colour palette ────────────────────────────────────────────────────────
    NAVY     = colors.HexColor("#0f172a")
    BLUE     = colors.HexColor("#1e3a8a")
    BLUE_L   = colors.HexColor("#3b82f6")
    SLATE    = colors.HexColor("#334155")
    SLATE_L  = colors.HexColor("#64748b")
    GOLD     = colors.HexColor("#d97706")
    GREEN    = colors.HexColor("#16a34a")
    RED      = colors.HexColor("#dc2626")
    GREY_BG  = colors.HexColor("#f8fafc")
    BORDER   = colors.HexColor("#e2e8f0")
    INNER    = colors.HexColor("#f1f5f9")
    HEAD_BG  = colors.HexColor("#dbeafe")
    HEAD_FG  = colors.HexColor("#1e40af")

    def __init__(self, output_path: str = "report.pdf"):
        self.output_path = output_path
        self.styles      = getSampleStyleSheet()
        self._build_styles()

    # ── Style definitions ─────────────────────────────────────────────────────

    def _build_styles(self):
        def _ps(name, **kw):
            return ParagraphStyle(name, **kw)

        self.s_cover_title = _ps("CoverTitle",
            fontName="Helvetica-Bold", fontSize=22, leading=28,
            textColor=self.NAVY, spaceAfter=6)

        self.s_cover_sub = _ps("CoverSub",
            fontName="Helvetica", fontSize=11, leading=15,
            textColor=self.SLATE_L, spaceAfter=12)

        self.s_section = _ps("Section",
            fontName="Helvetica-Bold", fontSize=13, leading=17,
            textColor=self.BLUE, spaceBefore=18, spaceAfter=8,
            keepWithNext=True)

        self.s_subsection = _ps("SubSection",
            fontName="Helvetica-Bold", fontSize=10.5, leading=14,
            textColor=self.NAVY, spaceBefore=10, spaceAfter=5,
            keepWithNext=True)

        self.s_body = _ps("Body",
            fontName="Helvetica", fontSize=9.5, leading=14,
            textColor=self.SLATE, spaceAfter=6)

        self.s_body_small = _ps("BodySmall",
            fontName="Helvetica", fontSize=8.5, leading=12,
            textColor=self.SLATE_L, spaceAfter=4)

        self.s_label = _ps("Label",
            fontName="Helvetica-Bold", fontSize=9, leading=12,
            textColor=self.NAVY)

        self.s_value = _ps("Value",
            fontName="Helvetica", fontSize=9, leading=12,
            textColor=self.SLATE)

        self.s_code = _ps("Code",
            fontName="Courier", fontSize=7.5, leading=11,
            textColor=self.SLATE, spaceAfter=4,
            backColor=colors.HexColor("#f1f5f9"))

        self.s_caption = _ps("Caption",
            fontName="Helvetica-Oblique", fontSize=8, leading=11,
            textColor=self.SLATE_L, spaceAfter=6, spaceBefore=2)

        self.s_highlight = _ps("Highlight",
            fontName="Helvetica-Bold", fontSize=9,
            textColor=colors.HexColor("#1d4ed8"))

    # ── Table style helpers ───────────────────────────────────────────────────

    def _meta_table_style(self):
        return TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), self.GREY_BG),
            ("BOX",          (0, 0), (-1, -1), 0.5, self.BORDER),
            ("INNERGRID",    (0, 0), (-1, -1), 0.5, self.INNER),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ])

    def _pred_table_style(self, n_rows):
        row_bgs = []
        for i in range(1, n_rows):
            bg = self.GREY_BG if i % 2 == 0 else colors.white
            row_bgs.append(("BACKGROUND", (0, i), (-1, i), bg))
        return TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), self.HEAD_BG),
            ("TEXTCOLOR",     (0, 0), (-1, 0), self.HEAD_FG),
            ("BOX",           (0, 0), (-1, -1), 0.6, colors.HexColor("#94a3b8")),
            ("GRID",          (0, 0), (-1, -1), 0.4, self.BORDER),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ] + row_bgs)

    # ── Section divider ───────────────────────────────────────────────────────

    def _section_divider(self, story, title: str):
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                color=self.BLUE, spaceAfter=4))
        story.append(Paragraph(title, self.s_section))

    # ── Figure block helper ───────────────────────────────────────────────────

    def _add_figure(self, story, path: str, width: float, height: float,
                    caption: str = ""):
        img = _safe_image(path, width, height)
        if img:
            story.append(img)
            if caption:
                story.append(Paragraph(f"<i>{caption}</i>", self.s_caption))
            story.append(Spacer(1, 8))

    # =========================================================================
    # generate_pdf  ─  main entry point
    # =========================================================================

    def generate_pdf(self, data: dict, visual_files: dict):
        """
        Assemble the full multi-page PDF.

        Args:
            data:         Analysis payload from Orchestrator.
            visual_files: Dict of generated image/HTML paths from VisualsGenerator.
        """
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=54, leftMargin=54,
            topMargin=80,   bottomMargin=70
        )

        story = []

        # ─── Resolve metadata ─────────────────────────────────────────────────
        meta = data.get("metadata", {})

        # Prefer data-level fields over metadata sub-dict
        protein_name = (meta.get("protein_name")
                        or data.get("protein_name")
                        or "Query Protein")
        # If still a hash-looking name, use a friendlier label
        if protein_name.startswith("seq_") or protein_name.startswith("motif_"):
            protein_name = "Query Protein (Novel Sequence)"

        gene_name    = meta.get("gene_name", "N/A")
        if gene_name in ("sequence", "motif", "sample", "N/A", ""):
            gene_name = "—"

        organism     = meta.get("organism", "N/A")
        function_txt = meta.get("function",
                                "Arginine methylation prediction via the ArgMe-XAI Agent.")
        uniprot_id   = data.get("uniprot_id", "N/A") or "N/A"
        ptm_type     = data.get("ptm_type", "methyl").capitalize()
        query_id     = data.get("query_id", "unknown")
        query_mode   = data.get("query_mode", "N/A")
        input_seq    = data.get("input_sequence", "")
        summary      = data.get("predictions_summary", {})
        positives    = data.get("positive_predictions", [])
        all_preds    = data.get("all_predictions", [])
        explanations = data.get("explanations", [])

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 1 — Cover / Header
        # ─────────────────────────────────────────────────────────────────────
        story.append(Paragraph("ProteinMind Clinical Report", self.s_cover_title))
        story.append(Paragraph(
            "Arginine Methylation Explainable AI Agent  ·  "
            "ArgMe-XAI Agent",
            self.s_cover_sub))
        story.append(Spacer(1, 10))

        # ── Metadata table ────────────────────────────────────────────────────
        meta_rows = [
            [Paragraph("Protein Name:", self.s_label),
             Paragraph(protein_name, self.s_value),
             Paragraph("UniProt ID:", self.s_label),
             Paragraph(uniprot_id, self.s_value)],
            [Paragraph("Gene Symbol:", self.s_label),
             Paragraph(gene_name, self.s_value),
             Paragraph("Organism:", self.s_label),
             Paragraph(organism, self.s_value)],
            [Paragraph("Target PTM:", self.s_label),
             Paragraph(ptm_type + "ylation", self.s_value),
             Paragraph("Analysis Date:", self.s_label),
             Paragraph(datetime.now().strftime("%Y-%m-%d"), self.s_value)],
            [Paragraph("Query Mode:", self.s_label),
             Paragraph(query_mode.capitalize(), self.s_value),
             Paragraph("Query ID:", self.s_label),
             Paragraph(query_id, self.s_value)],
        ]
        meta_tbl = Table(meta_rows, colWidths=[90, 160, 90, 160])
        meta_tbl.setStyle(self._meta_table_style())
        story.append(meta_tbl)
        story.append(Spacer(1, 12))

        # ── Prediction summary box ────────────────────────────────────────────
        self._section_divider(story, "1.  Prediction Summary")

        thr = summary.get("threshold_used", 0.5)
        thr_str = f"{thr:.3f}" if isinstance(thr, float) else str(thr)
        weights = summary.get("consensus_weights", {})

        summary_rows = [
            [Paragraph("Total R/K Sites Evaluated:", self.s_label),
             Paragraph(str(summary.get("total_sites_checked", "N/A")), self.s_value),
             Paragraph("Predicted Positive Sites:", self.s_label),
             Paragraph(str(summary.get("positive_sites_count", 0)), self.s_value)],
            [Paragraph("Decision Threshold:", self.s_label),
             Paragraph(thr_str, self.s_value),
             Paragraph("Fusion Mode:", self.s_label),
             Paragraph(summary.get("fusion_mode", "N/A"), self.s_value)],
        ]
        sum_tbl = Table(summary_rows, colWidths=[130, 120, 130, 120])
        sum_tbl.setStyle(self._meta_table_style())
        story.append(sum_tbl)
        story.append(Spacer(1, 6))

        # Consensus weights
        if weights:
            wt_text = "  ·  ".join(
                f"<b>{k.replace('P_', '')}:</b> {v:.2f}"
                for k, v in weights.items()
            )
            story.append(Paragraph(
                f"<b>Consensus Weights:</b>  {wt_text}", self.s_body_small))
        story.append(Spacer(1, 4))

        # ── Biological function ───────────────────────────────────────────────
        self._section_divider(story, "2.  Biological Function")
        story.append(Paragraph(function_txt, self.s_body))

        # ── Input sequence ────────────────────────────────────────────────────
        self._section_divider(story, "3.  Input Sequence")
        if input_seq:
            seq_len = len(input_seq)
            rk_count = sum(1 for c in input_seq if c in ("R", "K"))
            story.append(Paragraph(
                f"<b>Length:</b> {seq_len} aa  ·  "
                f"<b>R/K sites (candidate methylation targets):</b> {rk_count}",
                self.s_body))

            # Chunk sequence into 60-char lines with position labels
            chunk_size = 60
            seq_lines  = []
            for i in range(0, seq_len, chunk_size):
                chunk   = input_seq[i: i + chunk_size]
                pos_lbl = str(i + 1).rjust(6)
                seq_lines.append(f"{pos_lbl}   {chunk}")
            story.append(Paragraph("<br/>".join(seq_lines), self.s_code))
        else:
            story.append(Paragraph("Sequence not available in payload.", self.s_body))

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 2 — Predictions Table
        # ─────────────────────────────────────────────────────────────────────
        story.append(PageBreak())
        self._section_divider(story, "4.  Predicted Methylation Sites")

        if not positives:
            story.append(Paragraph(
                "No positive modification sites were identified above the "
                "validation threshold. This may indicate the sequence lacks "
                "known methylation motifs, or the confidence scores are "
                "uniformly below the decision boundary.",
                self.s_body))
        else:
            header = [
                Paragraph("<b>Pos.</b>",      self.s_label),
                Paragraph("<b>Res.</b>",       self.s_label),
                Paragraph("<b>Motif (11-mer)</b>", self.s_label),
                Paragraph("<b>P_final</b>",    self.s_label),
                Paragraph("<b>Confidence</b>", self.s_label),
                Paragraph("<b>Evidence Tier</b>", self.s_label),
                Paragraph("<b>Conserved In</b>",  self.s_label),
            ]
            table_data = [header]
            for p in positives:
                p_val  = p.get("P_final", p.get("probability", 0.0))
                conf   = p.get("confidence", "N/A")
                etier  = p.get("evidence_tier", "N/A")
                cons   = ", ".join(p.get("conserved_in", [])) or "—"
                motif  = p.get("motif", "N/A")

                # Colour-code confidence
                conf_style = self.s_value
                if conf == "High":
                    conf_style = ParagraphStyle("ConfHigh",
                        parent=self.s_value, textColor=self.GREEN)
                elif conf == "Moderate":
                    conf_style = ParagraphStyle("ConfMod",
                        parent=self.s_value, textColor=self.GOLD)
                elif conf in ("Low", "Very Low"):
                    conf_style = ParagraphStyle("ConfLow",
                        parent=self.s_value, textColor=self.RED)

                table_data.append([
                    Paragraph(str(p["position"]),       self.s_value),
                    Paragraph(p["residue"],             self.s_value),
                    Paragraph(f"<font face='Courier'>{motif}</font>", self.s_value),
                    Paragraph(f"{p_val:.4f}",           self.s_value),
                    Paragraph(conf,                     conf_style),
                    Paragraph(etier.upper(),            self.s_value),
                    Paragraph(cons,                     self.s_body_small),
                ])

            pred_tbl = Table(table_data,
                             colWidths=[36, 34, 120, 52, 60, 70, 82])
            pred_tbl.setStyle(self._pred_table_style(len(table_data)))
            story.append(pred_tbl)
        story.append(Spacer(1, 12))

        # ── All-sites score table (top 15) ────────────────────────────────────
        if all_preds:
            story.append(Paragraph("All Evaluated Sites (Top 15 by P_final)",
                                   self.s_subsection))
            top15 = sorted(all_preds,
                           key=lambda x: x.get("P_final", 0), reverse=True)[:15]
            h2 = [
                Paragraph("<b>Pos</b>",       self.s_label),
                Paragraph("<b>Res</b>",        self.s_label),
                Paragraph("<b>P_final</b>",   self.s_label),
                Paragraph("<b>ArgMe-XAI Score</b>",  self.s_label),
                Paragraph("<b>P_motif</b>",   self.s_label),
                Paragraph("<b>P_evidence</b>",self.s_label),
                Paragraph("<b>P_conserv</b>", self.s_label),
                Paragraph("<b>P_struct</b>",  self.s_label),
                Paragraph("<b>Label</b>",     self.s_label),
            ]
            t2_data = [h2]
            for s in top15:
                lbl = "✓ POS" if s.get("label", 0) == 1 else "— neg"
                t2_data.append([
                    Paragraph(str(s["position"]),                 self.s_body_small),
                    Paragraph(s["residue"],                       self.s_body_small),
                    Paragraph(f"{s.get('P_final', 0):.3f}",      self.s_body_small),
                    Paragraph(f"{s.get('P_hybrid', 0):.3f}",     self.s_body_small),
                    Paragraph(f"{s.get('P_motif', 0):.3f}",      self.s_body_small),
                    Paragraph(f"{s.get('P_evidence', 0):.3f}",   self.s_body_small),
                    Paragraph(f"{s.get('P_conserv', 0):.3f}",    self.s_body_small),
                    Paragraph(f"{s.get('P_struct', 0):.3f}",     self.s_body_small),
                    Paragraph(lbl,                                self.s_body_small),
                ])
            tbl2 = Table(t2_data,
                         colWidths=[33, 28, 50, 50, 48, 60, 55, 48, 44])
            tbl2.setStyle(self._pred_table_style(len(t2_data)))
            story.append(tbl2)

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 3 — 2D Structure + Sequence Annotation
        # ─────────────────────────────────────────────────────────────────────
        story.append(PageBreak())
        self._section_divider(story, "5.  Predicted 2D Protein Structure")
        story.append(Paragraph(
            "The diagram below shows a schematic 2D cartoon of the protein's "
            "predicted secondary structure (alpha-helices, beta-strands, loops) "
            "derived from sequence composition heuristics. "
            "Gold spikes indicate predicted methylation sites on R/K residues.",
            self.s_body))

        struct_2d = visual_files.get("structure_2d", "")
        self._add_figure(story, struct_2d, width=7.0 * inch, height=2.5 * inch,
                         caption="Fig. 1 — Predicted 2D secondary structure with methylation site annotation. "
                                 "Blue ribbons = α-helices, green arrows = β-strands, gold = methylation sites.")

        # ── 3D Structure PNG ──────────────────────────────────────────────────
        self._section_divider(story, "6.  3D Protein Backbone Structure")
        story.append(Paragraph(
            "The figure below shows a 3D backbone trace of the protein generated from "
            "sequence-based secondary structure prediction. The backbone is coloured "
            "by residue position (N-terminus blue → C-terminus red). "
            "Blue tubes = α-helices, green tubes = β-strands, "
            "thin line = loops/coil regions. "
            "<b>Gold spheres mark predicted arginine/lysine methylation sites.</b> "
            "The ★ (cyan) is the N-terminus; ■ (red) is the C-terminus.",
            self.s_body))

        struct_3d = visual_files.get("structure_3d", "")
        self._add_figure(story, struct_3d, width=6.5 * inch, height=5.2 * inch,
                         caption="Fig. 2 — 3D protein backbone trace with predicted methylation sites (gold spheres). "
                                 "Generated from sequence-based secondary structure heuristics. "
                                 "View angle: elevation 22°, azimuth 45°.")

        # ── Sequence annotation ───────────────────────────────────────────────
        self._section_divider(story, "7.  Full Sequence Methylation Site Map")
        story.append(Paragraph(
            "Linear sequence map showing all candidate R/K sites. Gold bars mark "
            "sites predicted positive by the ArgMe-XAI Agent consensus. "
            "Bar heights represent the final fusion score (P_final).",
            self.s_body))

        seq_ann = visual_files.get("sequence_annotation", "")
        self._add_figure(story, seq_ann, width=7.0 * inch, height=3.0 * inch,
                         caption="Fig. 2 — Sequence methylation site map. "
                                 "Gold = predicted methylated; grey = below threshold. "
                                 "Dashed red line = decision boundary at threshold.")

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 4 — Explainability (Attention + SHAP per site)
        # ─────────────────────────────────────────────────────────────────────
        story.append(PageBreak())
        self._section_divider(story,
            "8.  ArgMe-XAI Explainability — Attention &amp; SHAP Attribution")
        story.append(Paragraph(
            "For each top-predicted site, two interpretability views are shown: "
            "(i) the self-attention map capturing residue-level dependencies "
            "within the 11-mer window, and (ii) the occlusion-based SHAP attribution "
            "showing which flanking residues contribute most to the prediction. "
            "The center position (offset 0) is the target R/K residue.",
            self.s_body))

        attn_maps = visual_files.get("attention_maps", [])
        shap_plots = visual_files.get("shap_plots", [])

        if not attn_maps and not shap_plots:
            story.append(Paragraph(
                "No explainability figures were generated (no positive sites or "
                "untrained model in demo mode).", self.s_body))
        else:
            for idx, exp in enumerate(explanations):
                pos   = exp.get("position", "?")
                res   = exp.get("residue", "?")
                motif = exp.get("motif", "N/A")
                prob  = exp.get("probability", 0.0)

                story.append(KeepTogether([
                    Paragraph(
                        f"<b>Site  {res}{pos}</b>  ·  Motif: "
                        f"<font face='Courier'>{motif}</font>  ·  "
                        f"P_final = {prob:.4f}",
                        self.s_subsection),
                ]))

                # Side-by-side: attention | SHAP
                attn_img = (_safe_image(attn_maps[idx], 3.3 * inch, 2.8 * inch)
                            if idx < len(attn_maps) else None)
                shap_img = (_safe_image(shap_plots[idx], 3.3 * inch, 2.3 * inch)
                            if idx < len(shap_plots) else None)

                row = [attn_img or "", shap_img or ""]
                if attn_img or shap_img:
                    pair_tbl = Table([row], colWidths=[3.5 * inch, 3.5 * inch])
                    pair_tbl.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
                    ]))
                    story.append(pair_tbl)
                    story.append(Paragraph(
                        f"Left: self-attention map for {res}{pos}.  "
                        f"Right: Residue attribution (occlusion SHAP) for {res}{pos}.",
                        self.s_caption))
                story.append(Spacer(1, 12))

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 5 — Site Detail Cards
        # ─────────────────────────────────────────────────────────────────────
        site_cards = visual_files.get("site_cards", [])
        if site_cards:
            story.append(PageBreak())
            self._section_divider(story, "9.  Per-Site Agent Evidence Summary")
            story.append(Paragraph(
                "Each card below shows the five-agent evidence breakdown for one "
                "predicted methylation site. Bars show the score contribution from "
                "ArgMe-XAI, Motif, Structural, Conservation, and Dataset Evidence "
                "agents. The dashed line marks the 0.50 neutral baseline.",
                self.s_body))

            for card_path in site_cards:
                self._add_figure(story, card_path,
                                 width=6.8 * inch, height=2.8 * inch)
                story.append(Spacer(1, 6))

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 6 — Agent Score Heatmap + Radar Charts
        # ─────────────────────────────────────────────────────────────────────
        story.append(PageBreak())
        self._section_divider(story, "10.  Multi-Agent Evidence Score Heatmap")
        story.append(Paragraph(
            "The heatmap visualises all candidate sites (rows) scored by each "
            "of the five parallel agents (columns). Green = high evidence, "
            "red = low evidence. Sites where multiple agents agree (consistently "
            "high or low) provide the most reliable predictions.",
            self.s_body))

        heatmap = visual_files.get("agent_score_heatmap", "")
        self._add_figure(story, heatmap, width=7.0 * inch, height=4.0 * inch,
                         caption="Fig. 3 — Multi-agent evidence score heatmap. "
                                 "Rows = top-30 candidate sites sorted by P_final. "
                                 "Columns = 5 agents. Values shown in each cell.")

        # ── Radar charts ──────────────────────────────────────────────────────
        radar_charts = visual_files.get("radar_charts", [])
        if radar_charts:
            self._section_divider(story, "11.  Per-Site Agent Radar Charts")
            story.append(Paragraph(
                "Pentagon radar charts show the evidence profile for each predicted "
                "positive site. Larger filled area = stronger multi-agent consensus. "
                "Dots show individual agent scores.",
                self.s_body))

            # Lay out radars in a 2-column grid
            radar_pairs = []
            for i in range(0, len(radar_charts), 2):
                img_a = _safe_image(radar_charts[i], 3.2 * inch, 3.2 * inch)
                img_b = (_safe_image(radar_charts[i + 1], 3.2 * inch, 3.2 * inch)
                         if i + 1 < len(radar_charts) else "")
                radar_pairs.append([img_a or "", img_b or ""])

            if radar_pairs:
                radar_tbl = Table(radar_pairs, colWidths=[3.5 * inch, 3.5 * inch])
                radar_tbl.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING",    (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(radar_tbl)
                story.append(Paragraph(
                    "Fig. 4 — Per-site agent radar charts. "
                    "Each axis represents one of the five evidence agents.",
                    self.s_caption))

        # ─────────────────────────────────────────────────────────────────────
        # PAGE 7 — Disease Associations + Methodology
        # ─────────────────────────────────────────────────────────────────────
        story.append(PageBreak())
        self._section_divider(story, "12.  Pathology &amp; Disease Associations")
        story.append(Paragraph(
            "The chart below presents known or predicted disease associations "
            "for the identified methylation events. Where no database annotations "
            "exist, the sites are considered novel candidates.",
            self.s_body))

        dis_plot = visual_files.get("disease_plot", "")
        self._add_figure(story, dis_plot, width=6.5 * inch, height=3.0 * inch,
                         caption="Fig. 5 — Protein-disease association profile. "
                                 "Novel/unannotated sites are shown with a placeholder message.")

        # ── Literature ────────────────────────────────────────────────────────
        self._section_divider(story, "13.  Literature References")
        articles = data.get("pubmed_literature", [])
        if not articles:
            story.append(Paragraph(
                "This analysis is performed in dataset-centric mode. "
                "No external PubMed literature fetch was performed. "
                "All evidence is derived from the local training dataset "
                "(Human, Mouse, Rat, Chimpanzee methylation CSVs).",
                self.s_body))
        else:
            for art in articles:
                p_text = (f"<b>{art.get('title', 'No Title')}</b><br/>"
                          f"<i>Authors:</i> {art.get('authors', 'N/A')}<br/>"
                          f"<i>Journal:</i> {art.get('journal', 'N/A')} "
                          f"({art.get('year', 'N/A')})  |  PMID: "
                          f"<font color='blue'>{art.get('pmid', 'N/A')}</font>")
                story.append(Paragraph(p_text, self.s_body))
                story.append(Spacer(1, 6))

        # ── Methodology note ──────────────────────────────────────────────────
        self._section_divider(story, "14.  Methodology Notes")
        story.append(Paragraph(
            "<b>ArgMe-XAI Agent</b> — An explainable AI ensemble combining "
            "ESM-2 language model embeddings with attention-based sequence "
            "representations. Five parallel agents provide complementary "
            "evidence: (1) ArgMe-XAI sequence score, "
            "(2) motif sequence matching, (3) cross-species conservation, "
            "(4) dataset evidence (Hamming-similarity lookup), and "
            "(5) structural accessibility. Final predictions are fused by "
            "a ConsensusAgent using a weighted average or trained "
            "Logistic Regression meta-learner.",
            self.s_body))

        story.append(Paragraph(
            "<b>Dataset:</b> Human, Mouse, Rat, and Chimpanzee arginine/lysine "
            "methylation sequences from PhosphoSitePlus and UniMod. "
            "Positive and negative 11-mer windows are balanced per species. "
            "No external APIs are called during prediction — all evidence is "
            "derived from local CSV files.",
            self.s_body))

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"<i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  "
            f"ArgMe-XAI Agent  ·  M.Tech Thesis — Arginine Methylation Analysis</i>",
            self.s_body_small))

        # ─────────────────────────────────────────────────────────────────────
        # BUILD — NumberedCanvas adds header/footer to every page
        # ─────────────────────────────────────────────────────────────────────
        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"[Clinical Report] ✅ PDF built successfully → {self.output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rep = ClinicalReportGenerator("test_report.pdf")
    test_data = {
        "uniprot_id":    "P04637",
        "ptm_type":      "methyl",
        "query_id":      "seq_test01",
        "query_mode":    "sequence",
        "input_sequence": "MKRGGKPLRTAMKRGGKPLRTAMKRGGKPLRTA",
        "metadata": {
            "protein_name": "Tumor protein p53",
            "gene_name":    "TP53",
            "organism":     "Homo sapiens",
            "function":     "Acts as a tumor suppressor. Regulates target genes involved in cell cycle arrest, apoptosis, and DNA repair.",
        },
        "predictions_summary": {
            "total_sites_checked": 10,
            "positive_sites_count": 3,
            "threshold_used": 0.50,
            "fusion_mode": "weighted_average",
            "consensus_weights": {"P_hybrid": 0.45, "P_motif": 0.20, "P_evidence": 0.20, "P_conserv": 0.10, "P_struct": 0.05},
        },
        "positive_predictions": [
            {"position": 4,  "residue": "R", "motif": "MKRGGKPLRTA",
             "P_final": 0.78, "probability": 0.78, "confidence": "Moderate",
             "label": 1, "evidence_tier": "weak", "conserved_in": ["human"],
             "P_hybrid": 0.75, "P_motif": 0.82, "P_evidence": 0.45,
             "P_conserv": 0.35, "P_struct": 0.50},
        ],
        "all_predictions": [
            {"position": 4,  "residue": "R", "motif": "MKRGGKPLRTA",
             "P_final": 0.78, "label": 1, "P_hybrid": 0.75,
             "P_motif": 0.82, "P_evidence": 0.45, "P_conserv": 0.35, "P_struct": 0.50},
            {"position": 7,  "residue": "K", "motif": "GKPLRTAMKRG",
             "P_final": 0.32, "label": 0, "P_hybrid": 0.30,
             "P_motif": 0.25, "P_evidence": 0.10, "P_conserv": 0.10, "P_struct": 0.50},
        ],
        "explanations": [],
        "pubmed_literature": [],
    }
    rep.generate_pdf(test_data, {
        "attention_maps": [], "shap_plots": [], "disease_plot": "",
        "3d_viewer": "", "sequence_annotation": "", "agent_score_heatmap": "",
        "radar_charts": [], "structure_2d": "", "site_cards": [],
    })
    print("Test PDF written: test_report.pdf")

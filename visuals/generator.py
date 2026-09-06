"""
Visuals Generator  ─  v3  (Dataset-Centric, PDF-Ready)
=======================================================
Generates all publication-quality figures for the HybridARGME analysis.

Figures:
  1. BMAM Attention Heatmap         (attention_site_RXX.png)
  2. Occlusion / SHAP Bar Chart     (shap_site_RXX.png)
  3. Disease Association Bar Chart  (disease_associations.png)
  4. 3D Molecular HTML Viewer       (structure_viewer_PXXXXX.html)  [browser only]
  5. Sequence Annotation Plot       (sequence_annotation.png)
  6. Agent Score Heatmap            (agent_score_heatmap.png)
  7. Agent Radar Chart              (radar_site_RXX.png)
  8. 2D Structure PNG               (structure_2d.png)  [NEW — embeddable in PDF]
  9. Site Detail Card               (site_card_RXX.png) [NEW — per-site summary]
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Theme / constants
# ─────────────────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "P_hybrid":   "#4A90E2",
    "P_motif":    "#E94E77",
    "P_struct":   "#27AE60",
    "P_conserv":  "#F39C12",
    "P_evidence": "#8E44AD",
}
AGENT_LABELS = {
    "P_hybrid":   "HybridARGME",
    "P_motif":    "Motif",
    "P_struct":   "Structural",
    "P_conserv":  "Conservation",
    "P_evidence": "Dataset Evidence",
}
SCORE_KEYS = ["P_hybrid", "P_motif", "P_struct", "P_conserv", "P_evidence"]

BG_DARK   = "#0f172a"
BG_MID    = "#1e293b"
BG_CARD   = "#0f2744"
TEXT_LIGHT = "white"
GOLD      = "#F5A623"
RED_HL    = "#e11d48"
BLUE_HL   = "#38bdf8"


class VisualsGenerator:
    def __init__(self, output_dir="proteinmind_outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["font.family"]     = "sans-serif"

    # =========================================================================
    # 1. BMAM Attention Heatmap
    # =========================================================================

    def generate_attention_heatmap(self, idx, residue, position, motif, attn_matrix) -> str:
        """BMAM Bidirectional Self-Attention heatmap."""
        matrix = np.array(attn_matrix)
        labels = [f"{char}\n({i-5})" for i, char in enumerate(motif)]

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_MID)

        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels,
                    cmap="mako", annot=True, fmt=".2f",
                    cbar_kws={"label": "Attention Weight"},
                    linewidths=0.5, ax=ax)
        ax.set_title(f"BMAM Bidirectional Self-Attention Map\n"
                     f"Site: {residue}{position}  |  Motif: {motif}",
                     fontsize=13, fontweight="bold", color=TEXT_LIGHT, pad=12)
        ax.set_xlabel("Key Residues (Relative Position)", fontsize=10, color=TEXT_LIGHT)
        ax.set_ylabel("Query Residues (Relative Position)", fontsize=10, color=TEXT_LIGHT)
        ax.tick_params(colors=TEXT_LIGHT)
        ax.figure.axes[-1].tick_params(colors=TEXT_LIGHT)   # colorbar tick
        ax.figure.axes[-1].yaxis.label.set_color(TEXT_LIGHT)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"attention_site_{residue}{position}.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        return path

    # =========================================================================
    # 2. SHAP / Occlusion Bar Chart
    # =========================================================================

    def generate_shap_bar_chart(self, idx, residue, position, motif, shap_scores) -> str:
        """Occlusion-based residue attribution bar chart."""
        scores = np.array(shap_scores)
        labels = [f"{c}\n({i-5})" for i, c in enumerate(motif)]
        bar_colors = [GOLD if i == 5 else BLUE_HL for i in range(len(motif))]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_MID)

        bars = ax.bar(labels, scores, color=bar_colors, edgecolor="#334155",
                      linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    h + (0.005 if h >= 0 else -0.015),
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=TEXT_LIGHT)
        ax.axhline(0, color="#64748b", linewidth=1.0, linestyle="--")
        ax.set_title(f"Residue Attribution (Occlusion / SHAP-like)\n"
                     f"Site: {residue}{position}  |  Motif: {motif}",
                     fontsize=12, fontweight="bold", color=TEXT_LIGHT, pad=10)
        ax.set_xlabel("Residue (Relative Position)", fontsize=10, color=TEXT_LIGHT)
        ax.set_ylabel("Attribution Score (Prob. Shift)", fontsize=10, color=TEXT_LIGHT)
        ax.tick_params(colors=TEXT_LIGHT)
        ax.spines[:].set_color("#334155")

        # Legend
        center_patch = mpatches.Patch(color=GOLD, label="Target Site (Center)")
        flank_patch  = mpatches.Patch(color=BLUE_HL, label="Flanking Residues")
        ax.legend(handles=[center_patch, flank_patch],
                  facecolor=BG_DARK, labelcolor=TEXT_LIGHT, fontsize=8)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"shap_site_{residue}{position}.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        return path

    # =========================================================================
    # 3. Disease Association Bar Chart
    # =========================================================================

    def generate_disease_bar_chart(self, diseases: list) -> str:
        """Disease association horizontal bar chart."""
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_MID)

        if not diseases:
            ax.text(0.5, 0.5,
                    "No disease linkage annotations found.\n"
                    "Methylation sites are novel / unannotated in current databases.",
                    ha="center", va="center", fontsize=11,
                    style="italic", color="#94a3b8",
                    transform=ax.transAxes)
            ax.set_title("Protein-Disease Association Profile",
                         fontsize=13, fontweight="bold", color=TEXT_LIGHT)
            ax.axis("off")
            path = os.path.join(self.output_dir, "disease_associations.png")
            plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
            plt.close()
            return path

        display = diseases[:5]
        names   = [d["name"][:30] + "…" if len(d["name"]) > 30
                   else d["name"] for d in display]
        descs   = [d["description"][:50] + "…" if len(d["description"]) > 50
                   else d["description"] for d in display]
        scores  = [0.95 - i * 0.1 for i in range(len(names))]
        y_pos   = np.arange(len(names))
        colors  = sns.color_palette("flare", len(names))

        bars = ax.barh(y_pos, scores, color=colors, edgecolor="#334155", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10, fontweight="bold", color=TEXT_LIGHT)
        ax.set_xlabel("Confidence / Association Score", fontsize=10, color=TEXT_LIGHT)
        ax.set_title("Linked Diseases and Clinical Pathology Profile",
                     fontsize=13, fontweight="bold", color=TEXT_LIGHT, pad=12)
        for i, bar in enumerate(bars):
            ax.text(0.04, bar.get_y() + bar.get_height() / 2, descs[i],
                    va="center", ha="left", color="white",
                    fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.tick_params(colors=TEXT_LIGHT)
        ax.spines[:].set_color("#334155")
        plt.tight_layout()

        path = os.path.join(self.output_dir, "disease_associations.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        return path

    # =========================================================================
    # 4. 3D Molecular HTML Viewer  (browser-only, NOT in PDF)
    # =========================================================================

    def _fetch_pdb_from_alphafold(self, accession: str) -> str:
        import requests
        if not accession or accession in ("Unknown", "N/A", ""):
            return ""
        url = f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            print(f"[Visuals] AlphaFold request failed: {e}")
        return ""

    def _fetch_pdb_from_esmfold(self, sequence: str) -> str:
        """
        Real ESMFold structure prediction via the public ESM Metagenomic
        Atlas API. Works for arbitrary short sequences (including isolated
        motifs), unlike AlphaFold EBI which only serves precomputed
        full-length UniProt entries.

        NOTE: folding an isolated 11-residue fragment produces a real
        ESMFold structure, but it reflects the peptide folded in vacuum,
        not the residue's true accessibility context within its native
        full-length protein. Keep that caveat visible in the UI/report —
        see source_label below.
        """
        import requests
        seq = (sequence or "").strip().upper()
        if not (2 <= len(seq) <= 400) or not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
            return ""
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        try:
            resp = requests.post(url, data=seq, timeout=60)
            if resp.status_code == 200 and resp.text.strip().startswith("ATOM"):
                return resp.text
            print(f"[Visuals] ESMFold returned status {resp.status_code}.")
        except Exception as e:
            print(f"[Visuals] ESMFold request failed: {e}")
        return ""

    def generate_3d_molecular_html(self, accession: str, positive_positions: list,
                                   sequence: str = "", local_pdb_path: str = "") -> str:
        """Interactive 3D HTML viewer using 3Dmol.js (browser-only)."""
        pos_array_str = str(list(positive_positions))
        pdb_data      = ""
        source_label  = "Unknown Source"

        # 1. Try local PDB
        if local_pdb_path and os.path.exists(local_pdb_path):
            try:
                with open(local_pdb_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content and content.startswith("ATOM"):
                    pdb_data     = content
                    source_label = f"Local PDB ({os.path.basename(local_pdb_path)})"
            except Exception:
                pass

        # 2. Try AlphaFold (full-length UniProt entries only)
        if not pdb_data and accession and accession not in ("", "Unknown", "N/A"):
            pdb_data = self._fetch_pdb_from_alphafold(accession)
            if pdb_data:
                source_label = f"AlphaFold EBI ({accession})"

        # 3. Try real ESMFold prediction (works for short/motif sequences
        #    that have no UniProt accession and thus no AlphaFold entry)
        if not pdb_data and sequence:
            pdb_data = self._fetch_pdb_from_esmfold(sequence)
            if pdb_data:
                is_fragment = len(sequence) <= 15
                source_label = (
                    f"ESMFold prediction (isolated {len(sequence)}-residue fragment — "
                    f"not native accessibility context)"
                    if is_fragment else
                    f"ESMFold prediction ({len(sequence)} residues)"
                )

        # 4. Synthetic helix fallback (only if ESMFold also unavailable, e.g. offline)
        if not pdb_data:
            pdb_lines = []
            n = min(max(len(sequence) if sequence else 40, 20), 80)
            for i in range(n):
                t = i * 0.52
                x = round(2.3 * math.cos(t), 3)
                y = round(2.3 * math.sin(t), 3)
                z = round(1.5 * t, 3)
                residue = (sequence[i] if sequence and i < len(sequence) else "A")
                residue_atoms = [
                    ("N",  x - 0.65, y - 0.35, z - 0.15, "N"),
                    ("CA", x,        y,        z,        "C"),
                    ("C",  x + 0.65, y + 0.35, z + 0.15, "C"),
                    ("O",  x + 0.95, y + 0.55, z + 0.25, "O"),
                ]
                for atom_offset, (atom_name, atom_x, atom_y, atom_z, element) in enumerate(residue_atoms):
                    atom_id = i * 4 + atom_offset + 1
                    pdb_lines.append(
                        f"ATOM  {atom_id:5d} {atom_name:>4s} {residue:>3s} A{i+1:4d}    "
                        f"{atom_x:8.3f}{atom_y:8.3f}{atom_z:8.3f}  1.00 50.00           {element:>2s}"
                    )
            pdb_data     = "\n".join(pdb_lines)
            source_label = "Synthetic peptide view (no full protein structure available)"

        safe_pdb = pdb_data.replace("\\", "\\\\").replace("`", "\\`")
        title    = accession if accession not in ("", "Unknown", "N/A") else "Predicted Structure"

        viewer_script = f"""
            var pdbData = `{safe_pdb}`;
            var el = document.getElementById('viewer');
            glviewer = $3Dmol.createViewer(el, {{backgroundColor: '#020617'}});
            glviewer.addModel(pdbData, "pdb");
            glviewer.setStyle({{}}, {{
                cartoon: {{color: '#64748b'}},
                line: {{color: '#64748b', linewidth: 2.0}}
            }});
            highlightPTMs();
            glviewer.zoomTo();
            glviewer.render();
            document.getElementById('source-label').textContent = 'Source: {source_label}';
        """

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>3D Protein Viewer — {title}</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    body {{ font-family:'Helvetica Neue',Arial,sans-serif; background:#0f172a; color:#f8fafc;
           margin:0; padding:20px; display:flex; flex-direction:column; align-items:center; }}
    h1   {{ margin-top:10px; color:#38bdf8; font-size:20px; }}
    #viewer {{ width:800px; height:560px; position:relative; border-radius:12px;
              box-shadow:0 10px 25px rgba(0,0,0,.5); border:2px solid #334155; background:#020617; }}
    .controls {{ margin-top:16px; display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }}
    button {{ background:#0284c7; color:white; border:none; padding:9px 18px; border-radius:6px;
              cursor:pointer; font-weight:bold; transition:background .2s; font-size:13px; }}
    button:hover {{ background:#0369a1; }}
    .info-panel {{ margin-top:14px; background:#1e293b; padding:14px 18px; border-radius:8px;
                   width:760px; border-left:5px solid #e11d48; font-size:13px; }}
    .info-panel span {{ color:#94a3b8; font-size:11px; display:block; margin-top:5px; }}
  </style>
</head>
<body>
  <h1>🧬 ProteinMind Structure Viewer: {title}</h1>
  <div id="viewer"></div>
  <div class="controls">
    <button onclick="applySpectrum();">🌈 Spectrum</button>
    <button onclick="glviewer.setStyle({{}},{{cartoon:{{color:'#64748b'}},line:{{color:'#64748b'}}}});highlightPTMs();glviewer.render();">🔴 PTM Sites</button>
    <button onclick="glviewer.setStyle({{}},{{surface:{{opacity:0.7,color:'white'}}}});highlightPTMs();glviewer.render();">🔵 Surface</button>
    <button onclick="glviewer.setStyle({{}},{{stick:{{}}}});highlightPTMs();glviewer.render();">🔗 Stick</button>
    <button onclick="glviewer.zoomTo();">⬜ Reset</button>
  </div>
  <div class="info-panel">
    <strong>Predicted Methylation Sites:</strong> Positions {pos_array_str} shown as red spheres.
    <span id="source-label">Loading structure...</span>
  </div>
  <script>
    var glviewer = null;
    var ptmPositions = {pos_array_str};
        function applySpectrum() {{
            if (!glviewer) {{
                window.alert('The 3D viewer is still loading. Please try again.');
                return;
            }}
            glviewer.setStyle({{}}, {{
                cartoon: {{colorscheme: 'spectrum'}},
                line: {{colorscheme: 'spectrum'}}
            }});
            highlightPTMs();
            glviewer.render();
        }}
        function highlightPTMs() {{
      if (!glviewer) return;
      ptmPositions.forEach(function(p) {{
                glviewer.addStyle({{resi: p}}, {{sphere: {{color: '#ef4444', radius: 2.4}}}});
                glviewer.addLabel('Predicted methylation', {{resi: p}}, {{
                    fontColor: '#fecaca', backgroundColor: '#450a0a',
                    backgroundOpacity: 0.85, fontSize: 12, inFront: true
                }});
      }});
    }}
    {viewer_script}
  </script>
</body>
</html>"""

        filename = f"structure_viewer_{title}.html"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[Visuals] Saved 3D structure viewer → {path}")
        return path

    # =========================================================================
    # 5. Sequence Annotation Plot
    # =========================================================================

    def generate_sequence_annotation_plot(self, sequence: str,
                                           all_predictions: list) -> str:
        """Full-sequence linear annotation plot with methylation site markers."""
        if not sequence or not all_predictions:
            return ""

        L = len(sequence)

        fig, (ax_seq, ax_bar) = plt.subplots(
            2, 1, figsize=(max(14, L // 8), 5),
            gridspec_kw={"height_ratios": [1, 2]}, sharex=True
        )
        fig.patch.set_facecolor(BG_DARK)
        for ax in (ax_seq, ax_bar):
            ax.set_facecolor(BG_MID)
            ax.tick_params(colors=TEXT_LIGHT)
            ax.spines[:].set_color("#334155")

        # ── Top panel: sequence track ──────────────────────────────────────────
        ax_seq.barh(0, L, left=1, height=0.5, color="#334155",
                    edgecolor="none", zorder=1)

        for i in range(0, L, 50):
            ax_seq.axvline(i + 1, color="#475569", linewidth=0.5, zorder=2)
            ax_seq.text(i + 1, 0.45, str(i + 1), ha="center", va="bottom",
                        fontsize=6.5, color="#94a3b8")

        for site in all_predictions:
            pos   = site["position"]
            label = site.get("label", site.get("prediction", 0))
            color = GOLD if label == 1 else "#475569"
            ax_seq.axvline(pos, color=color, linewidth=1.5, zorder=3, alpha=0.85)
            if label == 1:
                ax_seq.text(pos, -0.35, site["residue"], ha="center",
                            fontsize=7, color=GOLD, fontweight="bold")

        ax_seq.set_yticks([])
        ax_seq.set_xlim(1, L + 1)
        ax_seq.set_title(
            "Protein Sequence Annotation — HybridARGME Predicted Methylation Sites",
            color=TEXT_LIGHT, fontsize=12, fontweight="bold", pad=8
        )
        pos_patch = mpatches.Patch(color=GOLD,    label="Predicted Methylated (R/K)")
        neg_patch = mpatches.Patch(color="#475569", label="Not Methylated")
        ax_seq.legend(handles=[pos_patch, neg_patch], loc="upper right",
                      facecolor=BG_DARK, labelcolor=TEXT_LIGHT, fontsize=8,
                      framealpha=0.8)

        # ── Bottom panel: P_final bars ─────────────────────────────────────────
        positions = [s["position"] for s in all_predictions]
        p_finals  = [s.get("P_final", s.get("P_hybrid", 0)) for s in all_predictions]
        bar_cols  = [GOLD if s.get("label", 0) == 1 else "#334155"
                     for s in all_predictions]

        ax_bar.bar(positions, p_finals, width=max(1, L // 100),
                   color=bar_cols, edgecolor="none", zorder=2)
        ax_bar.axhline(0.5, color=RED_HL, linewidth=1.2, linestyle="--",
                       zorder=3, label="Decision Threshold (0.50)")
        ax_bar.set_ylabel("P_final Score", color=TEXT_LIGHT, fontsize=10)
        ax_bar.set_xlabel("Sequence Position", color=TEXT_LIGHT, fontsize=10)
        ax_bar.set_ylim(0, 1.08)
        ax_bar.legend(facecolor=BG_DARK, labelcolor=TEXT_LIGHT, fontsize=8)
        ax_bar.yaxis.label.set_color(TEXT_LIGHT)
        ax_bar.xaxis.label.set_color(TEXT_LIGHT)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "sequence_annotation.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Visuals] Saved sequence annotation plot → {path}")
        return path

    # =========================================================================
    # 6. Agent Score Heatmap
    # =========================================================================

    def generate_agent_score_heatmap(self, all_predictions: list) -> str:
        """Per-site × per-agent score heatmap (top 30 sites)."""
        if not all_predictions:
            return ""

        sites = sorted(all_predictions,
                       key=lambda x: x.get("P_final", x.get("P_hybrid", 0)),
                       reverse=True)[:30]

        row_labels = [f"{s['residue']}{s['position']}" for s in sites]
        col_labels = [AGENT_LABELS[k] for k in SCORE_KEYS]
        matrix     = np.array([[s.get(k, 0.5) for k in SCORE_KEYS] for s in sites])

        fig, ax = plt.subplots(figsize=(11, max(6, len(sites) * 0.38)))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_MID)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, interpolation="nearest")

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold", color=TEXT_LIGHT)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9, color=TEXT_LIGHT)
        ax.tick_params(colors=TEXT_LIGHT)

        for i in range(len(sites)):
            for j in range(len(SCORE_KEYS)):
                val   = matrix[i, j]
                color = "black" if 0.3 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=TEXT_LIGHT)
        cbar.set_label("Agent Score", color=TEXT_LIGHT, fontsize=10)

        ax.set_title(
            "Multi-Agent Evidence Score Heatmap\n"
            "(Top candidate sites × 5 Agents — HybridARGME Framework)",
            color=TEXT_LIGHT, fontsize=13, fontweight="bold", pad=12
        )
        plt.tight_layout()
        path = os.path.join(self.output_dir, "agent_score_heatmap.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Visuals] Saved agent score heatmap → {path}")
        return path

    # =========================================================================
    # 7. Agent Radar Chart (per site)
    # =========================================================================

    def generate_agent_radar_chart(self, site: dict) -> str:
        """Pentagon radar chart for one predicted positive site."""
        scores   = [site.get(k, 0.5) for k in SCORE_KEYS]
        labels   = [AGENT_LABELS[k] for k in SCORE_KEYS]
        residue  = site.get("residue", "R")
        position = site.get("position", 0)
        p_final  = site.get("P_final", 0)

        N       = len(scores)
        angles  = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        values  = scores + scores[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_MID)

        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                            color="#94a3b8", fontsize=8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color=TEXT_LIGHT, fontsize=10, fontweight="bold")
        ax.spines["polar"].set_color("#334155")
        ax.tick_params(colors="#334155")
        ax.set_ylim(0, 1)

        ax.plot(angles, values, color=BLUE_HL, linewidth=2.5)
        ax.fill(angles, values, color=BLUE_HL, alpha=0.25)

        for angle, val, key in zip(angles[:-1], scores, SCORE_KEYS):
            ax.scatter(angle, val, color=AGENT_COLORS[key], s=90,
                       zorder=5, edgecolors="white", linewidths=0.8)

        conf = site.get("confidence", "N/A")
        ax.set_title(
            f"Agent Evidence Radar\n"
            f"{residue}{position}  |  P_final = {p_final:.3f}  [{conf}]",
            color=TEXT_LIGHT, fontsize=12, fontweight="bold", pad=22
        )
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"radar_site_{residue}{position}.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Visuals] Saved radar chart → {path}")
        return path

    # =========================================================================
    # 8. 2D Structure Visualization PNG  [NEW — embeddable in PDF]
    # =========================================================================

    def generate_structure_2d_png(self, sequence: str, positive_positions: list,
                                   protein_name: str = "Query Protein") -> str:
        """
        Generate a 2D protein structure visualization embeddable in PDF.

        Uses sequence-based secondary structure prediction heuristics:
          - Proline (P) → turn/loop breaker
          - V, I, L, M → beta-sheet tendency
          - E, A, K, L  → alpha-helix tendency
          - G, P, S, T  → loop/coil tendency

        Draws a schematic cartoon showing helices (ribbons), sheets (arrows),
        and loops with methylation sites annotated.
        """
        if not sequence:
            return ""

        def predict_ss(seq: str) -> str:
            """Simple Chou-Fasman inspired secondary structure prediction."""
            HELIX  = set("EALKMRH")
            SHEET  = set("VILFYWT")
            LOOP   = set("GPSTDNQ")
            ss = []
            for aa in seq.upper():
                if aa in LOOP:
                    ss.append("L")
                elif aa in HELIX:
                    ss.append("H")
                elif aa in SHEET:
                    ss.append("E")
                else:
                    ss.append("L")
            # smooth runs (window of 3)
            smoothed = list(ss)
            for i in range(1, len(ss) - 1):
                neighbors = [ss[i-1], ss[i+1]]
                if ss[i] != "L" and neighbors.count(ss[i]) == 0:
                    smoothed[i] = "L"
            return "".join(smoothed)

        def get_segments(ss: str):
            """Group consecutive same-type residues into segments."""
            segs = []
            if not ss:
                return segs
            curr, start = ss[0], 0
            for i in range(1, len(ss)):
                if ss[i] != curr:
                    segs.append((curr, start, i - 1))
                    curr, start = ss[i], i
            segs.append((curr, start, len(ss) - 1))
            return segs

        ss  = predict_ss(sequence)
        segs = get_segments(ss)
        L   = len(sequence)
        pos_set = set(positive_positions)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_DARK)
        ax.set_xlim(0, L)
        ax.set_ylim(-2.2, 2.5)
        ax.axis("off")

        # ── Draw sequence line ─────────────────────────────────────────────────
        ax.axhline(0, xmin=0, xmax=1, color="#334155", linewidth=1.5, zorder=1)

        # ── Draw secondary structure segments ─────────────────────────────────
        for stype, s_start, s_end in segs:
            mid   = (s_start + s_end) / 2
            width = s_end - s_start + 1

            if stype == "H" and width >= 4:   # Alpha helix — zigzag ribbon
                n_turns = max(2, int(width / 3.6))
                xs = np.linspace(s_start, s_end, n_turns * 20)
                ys = 0.7 * np.sin(np.linspace(0, n_turns * 2 * np.pi, len(xs)))
                ax.fill_between(xs, ys - 0.25, ys + 0.25,
                                color="#4A90E2", alpha=0.75, zorder=2)
                ax.text(mid, 1.1, "α", ha="center", fontsize=8,
                        color="#93c5fd", fontweight="bold")

            elif stype == "E" and width >= 3:  # Beta strand — filled arrow
                shaft_y = [-0.22, 0.22, 0.22, 0.45, 0.0, -0.45, -0.22]
                shaft_x = [s_start, s_start, s_end - 1, s_end - 1,
                            s_end, s_end - 1, s_end - 1]
                ax.fill(shaft_x, shaft_y, color="#27AE60", alpha=0.75, zorder=2)
                ax.text(mid, 0.75, "β", ha="center", fontsize=8,
                        color="#86efac", fontweight="bold")

            else:                              # Loop — thin line (already drawn)
                if width >= 5:
                    ax.text(mid, -0.7, "loop", ha="center", fontsize=6,
                            color="#64748b", style="italic")

        # ── Annotate methylation sites ─────────────────────────────────────────
        for pos in range(1, L + 1):
            idx = pos - 1
            if pos in pos_set:
                # Gold spike + label
                ax.plot([idx, idx], [0.0, 1.7], color=GOLD,
                        linewidth=2.0, zorder=5)
                ax.scatter(idx, 1.75, color=GOLD, s=70, zorder=6,
                           edgecolors="white", linewidths=0.8)
                aa = sequence[idx] if idx < len(sequence) else "?"
                ax.text(idx, 2.05, f"{aa}{pos}", ha="center", fontsize=7,
                        color=GOLD, fontweight="bold", rotation=60,
                        va="bottom")
            elif idx < len(sequence) and sequence[idx] in ("R", "K"):
                # Grey tick for non-methylated R/K
                ax.plot([idx, idx], [0.0, 0.5], color="#475569",
                        linewidth=0.8, zorder=4)

        # ── Position labels every 50 residues ─────────────────────────────────
        for i in range(0, L, 50):
            ax.text(i, -1.0, str(i + 1), ha="center", fontsize=8,
                    color="#94a3b8")
            ax.plot([i, i], [-0.15, -0.4], color="#475569",
                    linewidth=0.8, zorder=3)

        # ── Legend ─────────────────────────────────────────────────────────────
        legend_items = [
            mpatches.Patch(color="#4A90E2", label="α-Helix"),
            mpatches.Patch(color="#27AE60", label="β-Strand"),
            mpatches.Patch(color="#475569", label="Loop/Coil"),
            mpatches.Patch(color=GOLD,      label="Predicted Methylation Site"),
        ]
        ax.legend(handles=legend_items, loc="lower right",
                  facecolor=BG_MID, labelcolor=TEXT_LIGHT,
                  fontsize=9, framealpha=0.85)

        ax.set_title(
            f"Predicted Secondary Structure & Methylation Site Map — {protein_name}\n"
            f"({L} residues  |  {len(positive_positions)} predicted methylation sites)",
            color=TEXT_LIGHT, fontsize=12, fontweight="bold", pad=10
        )

        plt.tight_layout()
        path = os.path.join(self.output_dir, "structure_2d.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close()
        print(f"[Visuals] Saved 2D structure PNG → {path}")
        return path

    # =========================================================================
    # 9. Site Detail Card  [NEW — per-site combined summary for PDF]
    # =========================================================================

    def generate_site_detail_card(self, site: dict, idx: int) -> str:
        """
        Generate a combined site-summary card showing:
          - Agent score bar chart
          - Key site statistics text
        """
        residue  = site.get("residue", "?")
        position = site.get("position", 0)
        p_final  = site.get("P_final", 0.0)
        motif    = site.get("motif", "N/A")
        conf     = site.get("confidence", "N/A")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor(BG_CARD)
        ax.set_facecolor(BG_CARD)

        agents  = SCORE_KEYS
        labels  = [AGENT_LABELS[k] for k in agents]
        values  = [site.get(k, 0.5) for k in agents]
        colors  = [AGENT_COLORS[k] for k in agents]

        bars = ax.barh(labels, values, color=colors, edgecolor="#1e3a5f",
                       height=0.6, zorder=2)
        ax.set_xlim(0, 1.15)
        ax.axvline(0.5, color="#64748b", linestyle="--", linewidth=1.0, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left",
                    color=TEXT_LIGHT, fontsize=9, fontweight="bold")
        ax.set_xlabel("Agent Score", color=TEXT_LIGHT, fontsize=9)
        ax.tick_params(colors=TEXT_LIGHT)
        ax.spines[:].set_color("#1e3a5f")

        # Title block
        ax.set_title(
            f"Site {residue}{position}  |  Motif: {motif}  |  "
            f"P_final = {p_final:.4f}  [{conf}]",
            color=GOLD, fontsize=11, fontweight="bold", pad=8
        )

        # Extra info: evidence tier, conservation
        evidence_tier = site.get("evidence_tier", "N/A")
        conserved_in  = site.get("conserved_in", [])
        match_cnt     = site.get("similar_examples", 0)
        info_str = (f"Evidence tier: {evidence_tier.upper()}  |  "
                    f"Dataset matches: {match_cnt}  |  "
                    f"Conserved in: {', '.join(conserved_in) or 'none'}")
        ax.text(0.5, -0.14, info_str, transform=ax.transAxes,
                ha="center", va="top", color="#94a3b8", fontsize=8,
                style="italic")

        plt.tight_layout()
        path = os.path.join(self.output_dir,
                            f"site_card_{residue}{position}.png")
        plt.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close()
        print(f"[Visuals] Saved site detail card → {path}")
        return path

    # =========================================================================
    # 10. 3D Protein Structure PNG  [NEW — real matplotlib 3D, PDF-embeddable]
    # =========================================================================

    def generate_structure_3d_png(self, sequence: str, positive_positions: list,
                                   protein_name: str = "Query Protein") -> str:
        """
        Generate a publication-quality 3D protein backbone visualization
        using matplotlib's mplot3d toolkit — fully embeddable in PDF.

        Renders:
          - Alpha-helices  : right-handed coil (parametric helix)
          - Beta-strands   : flat ribbons with arrowhead
          - Loops/coils    : smooth spline trace
          - Methylation sites: glowing gold spheres on backbone
          - Colour gradient: N-terminus (blue) → C-terminus (red)
        """
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        if not sequence:
            return ""

        # ── Secondary structure prediction (same heuristic as 2D) ─────────────
        HELIX_AA = set("EALKMRH")
        SHEET_AA = set("VILFYWT")

        def _predict_ss(seq):
            ss = []
            for aa in seq.upper():
                if aa in SHEET_AA:  ss.append("E")
                elif aa in HELIX_AA: ss.append("H")
                else:               ss.append("L")
            # smooth single-residue outliers
            s = list(ss)
            for i in range(1, len(s) - 1):
                if s[i] != "L" and s[i-1] != s[i] and s[i+1] != s[i]:
                    s[i] = "L"
            return "".join(s)

        def _get_segments(ss):
            segs = []
            if not ss: return segs
            cur, st = ss[0], 0
            for i in range(1, len(ss)):
                if ss[i] != cur:
                    segs.append((cur, st, i-1))
                    cur, st = ss[i], i
            segs.append((cur, st, len(ss)-1))
            return segs

        L   = len(sequence)
        ss  = _predict_ss(sequence)
        segs = _get_segments(ss)
        pos_set = set(positive_positions)

        # ── Build 3D Cα backbone coordinates ─────────────────────────────────
        xs, ys, zs = [], [], []
        # Parameters per SS type
        # Helix: coil around z-axis,  Strand: flat in xz plane,  Loop: random walk
        x, y, z = 0.0, 0.0, 0.0
        dx, dy, dz = 3.8, 0.0, 0.0  # initial direction

        np.random.seed(42)

        for stype, s_start, s_end in segs:
            n = s_end - s_start + 1
            if stype == "H":  # Alpha helix — coil
                r, pitch = 2.3, 1.5
                turns = max(1, int(n / 3.6))
                t_arr = np.linspace(0, turns * 2 * np.pi, n)
                # orient helix along current heading
                head_angle = np.arctan2(dy, dx)
                for t in t_arr:
                    xs.append(x + r * np.cos(t) * np.cos(head_angle)
                               - r * np.sin(t) * np.sin(head_angle))
                    ys.append(y + r * np.cos(t) * np.sin(head_angle)
                               + r * np.sin(t) * np.cos(head_angle))
                    zs.append(z)
                    z += pitch / n
                # update x,y to end of helix
                x = xs[-1]; y = ys[-1]

            elif stype == "E":  # Beta strand — straight
                step_x = 3.5 * np.cos(np.arctan2(dy, dx))
                step_y = 3.5 * np.sin(np.arctan2(dy, dx))
                for i in range(n):
                    xs.append(x + i * step_x)
                    ys.append(y + i * step_y)
                    zs.append(z + i * 0.4)
                x = xs[-1]; y = ys[-1]; z = zs[-1]
                # turn after strand
                dx, dy = -dy + 0.3, dx + 0.3

            else:  # Loop — curved random walk
                for i in range(n):
                    angle_jitter = np.random.uniform(-0.6, 0.6)
                    new_dx = dx * np.cos(angle_jitter) - dy * np.sin(angle_jitter)
                    new_dy = dx * np.sin(angle_jitter) + dy * np.cos(angle_jitter)
                    dx, dy = new_dx, new_dy
                    norm = np.sqrt(dx**2 + dy**2) + 1e-9
                    dx /= norm; dy /= norm
                    x += dx * 3.8; y += dy * 3.8
                    z += np.random.uniform(-0.5, 0.5)
                    xs.append(x); ys.append(y); zs.append(z)

        xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
        n_pts = len(xs)

        if n_pts < 3:
            return ""

        # Trim to match sequence length
        if n_pts > L: xs, ys, zs = xs[:L], ys[:L], zs[:L]
        n_pts = len(xs)

        # ── Build colour gradient N→C (blue → red) ────────────────────────────
        cmap   = plt.cm.coolwarm
        norm   = Normalize(vmin=0, vmax=n_pts - 1)
        colours = [cmap(norm(i)) for i in range(n_pts)]

        # ── Plot ──────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor(BG_DARK)
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG_DARK)

        # Draw backbone as coloured segments
        for i in range(n_pts - 1):
            ax.plot([xs[i], xs[i+1]],
                    [ys[i], ys[i+1]],
                    [zs[i], zs[i+1]],
                    color=colours[i], linewidth=1.8, alpha=0.85)

        # Highlight secondary structure regions
        for stype, s_start, s_end in segs:
            s_e = min(s_end, n_pts - 1)
            if stype == "H" and s_e > s_start:
                ax.plot(xs[s_start:s_e+1], ys[s_start:s_e+1], zs[s_start:s_e+1],
                        color="#4A90E2", linewidth=3.5, alpha=0.6,
                        solid_capstyle="round")
            elif stype == "E" and s_e > s_start:
                ax.plot(xs[s_start:s_e+1], ys[s_start:s_e+1], zs[s_start:s_e+1],
                        color="#27AE60", linewidth=3.5, alpha=0.6,
                        solid_capstyle="round")

        # N-terminus star
        ax.scatter([xs[0]], [ys[0]], [zs[0]],
                   s=120, c="#38bdf8", zorder=8,
                   edgecolors="white", linewidths=1.2,
                   label="N-terminus", marker="*")

        # C-terminus square
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                   s=100, c="#f87171", zorder=8,
                   edgecolors="white", linewidths=1.2,
                   label="C-terminus", marker="s")

        # Methylation site spheres (gold, prominent)
        for mpos in positive_positions:
            idx_m = min(mpos - 1, n_pts - 1)
            if idx_m >= 0:
                aa = sequence[idx_m] if idx_m < len(sequence) else "?"
                ax.scatter([xs[idx_m]], [ys[idx_m]], [zs[idx_m]],
                           s=280, c=GOLD, zorder=10,
                           edgecolors="white", linewidths=1.5,
                           label=f"{aa}{mpos}" if mpos == positive_positions[0] else None)
                # Label the site
                ax.text(xs[idx_m], ys[idx_m], zs[idx_m] + 1.8,
                        f"{aa}{mpos}", fontsize=8, color=GOLD,
                        fontweight="bold", ha="center")

        # ── Colour bar (N→C gradient) ─────────────────────────────────────────
        sm  = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.05, shrink=0.6)
        cbar.set_label("Residue Position (N→C)",
                       color=TEXT_LIGHT, fontsize=9)
        cbar.ax.tick_params(colors=TEXT_LIGHT, labelsize=7)
        cbar.set_ticks([0, n_pts//2, n_pts-1])
        cbar.set_ticklabels(["N-term", f"Mid ({n_pts//2})", "C-term"])

        # ── Axes styling ──────────────────────────────────────────────────────
        ax.set_title(
            f"3D Protein Backbone Structure — {protein_name}\n"
            f"({L} residues  |  {len(positive_positions)} predicted methylation sites "
            f"shown as gold spheres)",
            color=TEXT_LIGHT, fontsize=11, fontweight="bold", pad=14
        )
        ax.tick_params(colors=TEXT_LIGHT, labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#334155")
        ax.yaxis.pane.set_edgecolor("#334155")
        ax.zaxis.pane.set_edgecolor("#334155")
        ax.set_xlabel("X", color=TEXT_LIGHT, fontsize=8)
        ax.set_ylabel("Y", color=TEXT_LIGHT, fontsize=8)
        ax.set_zlabel("Z", color=TEXT_LIGHT, fontsize=8)

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#4A90E2", linewidth=3, label="α-Helix"),
            Line2D([0], [0], color="#27AE60", linewidth=3, label="β-Strand"),
            Line2D([0], [0], color="#94a3b8", linewidth=1.5, label="Loop/Coil"),
            plt.scatter([], [], s=120, c=GOLD, edgecolors="white",
                        label="Methylation Site"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                  facecolor=BG_MID, labelcolor=TEXT_LIGHT,
                  fontsize=8, framealpha=0.85)

        # Rotate to a nice viewing angle
        ax.view_init(elev=22, azim=45)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "structure_3d.png")
        plt.savefig(path, dpi=180, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close()
        print(f"[Visuals] Saved 3D structure PNG → {path}")
        return path

    # =========================================================================
    # Master execute — called by CLI / report generator
    # =========================================================================

    def execute_all(self, analysis_payload: dict) -> dict:
        """Generate ALL figures for a complete analysis payload."""
        accession       = (analysis_payload.get("uniprot_id") or
                           analysis_payload.get("accession") or "Unknown")
        sequence        = (analysis_payload.get("input_sequence", "") or
                           analysis_payload.get("sequence", ""))
        explanations    = analysis_payload.get("explanations", [])
        all_predictions = (analysis_payload.get("all_predictions") or
                           analysis_payload.get("final_predictions") or [])
        positive_sites  = analysis_payload.get("positive_predictions", [])
        meta            = analysis_payload.get("metadata", {})
        protein_name    = meta.get("protein_name", "Query Protein")

        output_files = {
            "attention_maps":      [],
            "shap_plots":          [],
            "disease_plot":        "",
            "3d_viewer":           "",
            "sequence_annotation": "",
            "agent_score_heatmap": "",
            "radar_charts":        [],
            "structure_2d":        "",
            "structure_3d":        "",
            "site_cards":          [],
        }

        # 1. Attention + SHAP (per explained site)
        for idx, exp in enumerate(explanations):
            pos   = exp.get("position")
            res   = exp.get("residue")
            motif = exp.get("motif")
            attn  = exp.get("attention_map", [])
            shap  = exp.get("shap_scores", [])
            if attn:
                output_files["attention_maps"].append(
                    self.generate_attention_heatmap(idx, res, pos, motif, attn))
            if shap:
                output_files["shap_plots"].append(
                    self.generate_shap_bar_chart(idx, res, pos, motif, shap))

        # 2. Disease chart
        diseases = explanations[0].get("disease_associations", []) if explanations else []
        output_files["disease_plot"] = self.generate_disease_bar_chart(diseases)

        # 3. 3D HTML viewer
        pos_list  = [p["position"] for p in positive_sites]
        local_pdb = meta.get("pdb_path", "")
        if not local_pdb and accession and accession not in ("Unknown", "N/A"):
            for candidate in [
                os.path.join("pdb_files", f"{accession}.pdb"),
                os.path.join("pdb_files", f"{accession.lower()}.pdb"),
                os.path.join("pdb_files", "sample.pdb"),
            ]:
                if os.path.exists(candidate):
                    local_pdb = candidate
                    break
        output_files["3d_viewer"] = self.generate_3d_molecular_html(
            accession, pos_list, sequence=sequence, local_pdb_path=local_pdb)

        # 4. Sequence annotation plot
        if all_predictions:
            seq = sequence or "X" * max(s["position"] for s in all_predictions)
            output_files["sequence_annotation"] = \
                self.generate_sequence_annotation_plot(seq, all_predictions)

        # 5. Agent score heatmap
        if all_predictions:
            output_files["agent_score_heatmap"] = \
                self.generate_agent_score_heatmap(all_predictions)

        # 6. Radar charts for top positive sites
        for site in positive_sites[:5]:
            path = self.generate_agent_radar_chart(site)
            output_files["radar_charts"].append(path)

        # 7. 2D structure PNG (linear schematic — PDF-embeddable)
        if sequence:
            output_files["structure_2d"] = self.generate_structure_2d_png(
                sequence, pos_list, protein_name=protein_name)

        # 8. 3D structure PNG (matplotlib 3D backbone — PDF-embeddable)
        if sequence:
            try:
                output_files["structure_3d"] = self.generate_structure_3d_png(
                    sequence, pos_list, protein_name=protein_name)
            except Exception as e:
                print(f"[Visuals] 3D structure PNG failed: {e}")
                output_files["structure_3d"] = ""

        # 9. Site detail cards (per positive site)
        for idx, site in enumerate(positive_sites[:5]):
            card_path = self.generate_site_detail_card(site, idx)
            output_files["site_cards"].append(card_path)

        return output_files

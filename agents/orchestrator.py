"""
Orchestrator  ─  Dataset-Centric Multi-Agent Pipeline
=====================================================
Coordinates all agents to produce the final prediction report.

Pipeline (v4 — Dataset-Centric Agentic, Local-First):
  1. Parse query → extract sequence, motif, or sample ID (NO UniProt)
  2. HybridARGMEAgent       → P_hybrid for every R/K site
  3. MotifAgent             → P_motif  (parallel)
  4. ConservationAgent      → P_conserv(parallel, dataset-based only)
  5. DatasetEvidenceAgent   → P_evidence (parallel, replaces P_lit)
  6. StructuralAgent        → P_struct  (neutral 0.50, no accession needed)
  7. ConsensusAgent         → P_final   (fused label + confidence)
  8. ExplainerAgent         → attention maps + SHAP scores

Accepted query formats:
  - "Analyze motif: FTLQIRGRERF"          → direct 11-mer
  - "Analyze sequence: MKRGGKPLRTA..."    → full protein sequence
    - "Analyze P04637 for methylation"       → UniProt accession lookup
  - "Analyze sample #125 from dataset"    → row lookup by index

All 4 scoring agents (steps 3-6) run on EVERY candidate site in parallel
using Python ThreadPoolExecutor for speed.

UniProt / PubMed dependencies: REMOVED.
"""

import re
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.prediction_agent        import HybridARGMEAgent
from agents.motif_agent             import MotifAgent
from agents.structural_agent        import StructuralAgent
from agents.conservation_agent      import ConservationAgent
from agents.dataset_evidence_agent  import DatasetEvidenceAgent
from agents.consensus_agent         import ConsensusAgent
from agents.explainer_agent         import ExplainerAgent


# Standard amino acid alphabet for sequence validation
_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Default local data-dir candidates (tried in order)
_DEFAULT_DATA_DIRS = [
    "data-11",
    os.path.join(os.path.dirname(__file__), "data-11"),
    "/content/drive/MyDrive/ProteinMind_Agent/data-11",
    "/kaggle/working/ProteinMind_Agent/data-11",
]


def _resolve_data_dir(data_dir: str = None) -> str:
    """Resolve the dataset directory — local first, Colab fallback."""
    candidates = ([data_dir] if data_dir else []) + _DEFAULT_DATA_DIRS
    for c in candidates:
        if c and os.path.exists(c):
            return c
    # Return the first non-None candidate even if it doesn't exist
    # (agents will handle the empty-load gracefully)
    return data_dir or _DEFAULT_DATA_DIRS[0]


class Orchestrator:
    """
    Multi-Agent Orchestrator for HybridARGME protein methylation prediction.
    Fully dataset-centric — no external API calls required.
    """

    def __init__(self, checkpoint_path=None, consensus_model_path=None,
                 data_dir=None):
        print("[Orchestrator] Initializing Dataset-Centric Agentic HybridARGME Framework...")

        resolved_data_dir = _resolve_data_dir(data_dir)
        print(f"[Orchestrator] Dataset directory resolved: {os.path.abspath(resolved_data_dir)}")

        self.hybridargme_agent      = HybridARGMEAgent(checkpoint_path)
        self.motif_agent            = MotifAgent()
        self.structural_agent       = StructuralAgent()
        self.conservation_agent     = ConservationAgent(data_dir=resolved_data_dir)
        self.dataset_evidence_agent = DatasetEvidenceAgent(data_dir=resolved_data_dir)
        self.consensus_agent        = ConsensusAgent(consensus_model_path)
        self.explainer_agent        = ExplainerAgent(self.hybridargme_agent)

        print("[Orchestrator] All 7 agents initialized successfully.\n")

    # ── Query Parsing ─────────────────────────────────────────────────────────

    def parse_query(self, query: str) -> dict:
        """
        Parse the user query into a structured input dict.

        Supported formats:
          1. "Analyze motif: FTLQIRGRERF"
             → {"mode": "motif", "sequence": "FTLQIRGRERF", "query_id": <hash>}

          2. "Analyze sequence: MKRGGKPLRTA..."
             → {"mode": "sequence", "sequence": "MKRGGKPLRTA...", "query_id": <hash>}

          3. "Analyze sample #125 from dataset" / "sample 125"
             → {"mode": "sample", "sample_index": 125, "query_id": "sample_125"}

          4. Bare amino acid string (≥5 chars, all AA alphabet)
             → {"mode": "sequence", "sequence": ..., "query_id": <hash>}

        Returns dict with parsed result, or {"error": "..."} on failure.
        """
        query = query.strip()

        # ── Format 1: motif keyword ────────────────────────────────────────────
        motif_match = re.search(
            r"motif\s*[:\-]?\s*([A-Z]{5,15})", query, re.IGNORECASE
        )
        if motif_match:
            seq = motif_match.group(1).upper()
            return {
                "mode":     "motif",
                "sequence": seq,
                "query_id": f"motif_{seq[:6]}",
            }

        # ── Format 2: sequence keyword ─────────────────────────────────────────
        seq_match = re.search(
            r"sequence\s*[:\-]?\s*([A-Z]{5,})", query, re.IGNORECASE
        )
        if seq_match:
            seq = seq_match.group(1).upper()
            qid = "seq_" + hashlib.md5(seq.encode()).hexdigest()[:8]
            return {
                "mode":     "sequence",
                "sequence": seq,
                "query_id": qid,
            }

        # ── Format 3: sample index ─────────────────────────────────────────────
        sample_match = re.search(
            r"sample\s*[#\-]?\s*(\d+)", query, re.IGNORECASE
        )
        if sample_match:
            idx = int(sample_match.group(1))
            return {
                "mode":         "sample",
                "sample_index": idx,
                "query_id":     f"sample_{idx}",
            }

        # ── Format 4: UniProt accession ────────────────────────────────────────
        accession_match = re.search(
            r"\b([A-NP-Z][0-9][A-Z0-9]{3}[0-9])\b", query, re.IGNORECASE
        )
        if accession_match:
            accession = accession_match.group(1).upper()
            return {
                "mode":       "accession",
                "accession":  accession,
                "query_id":   accession,
            }

        # ── Format 5: bare amino acid string ──────────────────────────────────
        bare = re.sub(r"\s+", "", query).upper()
        if len(bare) >= 5 and all(c in _AA_SET for c in bare):
            qid = "seq_" + hashlib.md5(bare.encode()).hexdigest()[:8]
            return {
                "mode":     "sequence",
                "sequence": bare,
                "query_id": qid,
            }

        return {
            "error": (
                "Could not parse query. Use one of:\n"
                "  • 'Analyze motif: FTLQIRGRERF'\n"
                "  • 'Analyze sequence: MKRGGKPLRTA'\n"
                "  • 'Analyze sample #125 from dataset'\n"
                "  • Or provide a bare amino acid sequence."
            )
        }

    def _resolve_sequence(self, parsed: dict) -> str:
        """
        Resolve the final amino acid sequence from a parsed query dict.

        For 'sample' mode, look up the sequence from the DatasetEvidenceAgent
        dataset (which is already loaded in memory).
        """
        mode = parsed.get("mode")

        if mode in ("motif", "sequence"):
            return parsed["sequence"]

        if mode == "accession":
            import requests

            accession = parsed["accession"]
            url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                fasta_lines = response.text.splitlines()
                sequence = "".join(
                    line.strip() for line in fasta_lines
                    if line.strip() and not line.startswith(">")
                ).upper()
                if sequence and all(residue in _AA_SET for residue in sequence):
                    return sequence
            except requests.RequestException as exc:
                print(f"[Orchestrator] UniProt lookup failed for {accession}: {exc}")
            return ""

        if mode == "sample":
            idx = parsed.get("sample_index", 0)
            dataset = self.dataset_evidence_agent.dataset
            if not dataset:
                return ""
            if idx < 0 or idx >= len(dataset):
                print(f"[Orchestrator] Sample index {idx} out of range "
                      f"(dataset has {len(dataset)} entries).")
                return ""
            return dataset[idx]["motif"]   # 11-mer from dataset

        return ""

    # ── Parallel Scoring ──────────────────────────────────────────────────────

    def _run_parallel_scoring(self, sites: list, accession: str = "") -> list:
        """
        Run MotifAgent, ConservationAgent, DatasetEvidenceAgent, and
        StructuralAgent in parallel threads.

        Returns merged site list with P_motif, P_conserv, P_evidence,
        P_struct added.
        """

        def run_motif():
            return self.motif_agent.score_all_sites(sites)

        def run_conservation():
            return self.conservation_agent.score_all_sites(sites)

        def run_dataset_evidence():
            return self.dataset_evidence_agent.score_all_sites(sites)

        def run_structural():
            # No accession = dataset-centric neutral mode (no API call)
            return self.structural_agent.score_all_sites(sites, accession=accession)

        print("[Orchestrator] Running Motif / Conservation / DatasetEvidence / "
              "Structural agents in parallel...")

        motif_results = conserv_results = evidence_results = struct_results = None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_motif):            "motif",
                executor.submit(run_conservation):     "conservation",
                executor.submit(run_dataset_evidence): "dataset_evidence",
                executor.submit(run_structural):       "structural",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "motif":
                        motif_results = result
                    elif name == "conservation":
                        conserv_results = result
                    elif name == "dataset_evidence":
                        evidence_results = result
                    elif name == "structural":
                        struct_results = result
                    print(f"[Orchestrator]   ✓ {name.replace('_', ' ').title()} Agent done.")
                except Exception as e:
                    print(f"[Orchestrator]   ✗ {name} Agent error: {e}")

        # ── Merge all scores ───────────────────────────────────────────────────
        def index_by_pos(site_list):
            return {s["position"]: s for s in site_list} if site_list else {}

        motif_idx    = index_by_pos(motif_results)
        conserv_idx  = index_by_pos(conserv_results)
        evidence_idx = index_by_pos(evidence_results)
        struct_idx   = index_by_pos(struct_results)

        merged = []
        for site in sites:
            pos      = site["position"]
            enriched = dict(site)

            # Motif scores
            enriched["P_motif"]       = motif_idx.get(pos, {}).get("P_motif", 0.50)
            enriched["matched_motifs"] = motif_idx.get(pos, {}).get("matched_motifs", [])
            enriched["physical_properties"] = motif_idx.get(pos, {}).get("physical_properties", {})

            # Conservation scores
            enriched["P_conserv"]    = conserv_idx.get(pos, {}).get("P_conserv", 0.25)
            enriched["conserved_in"] = conserv_idx.get(pos, {}).get("conserved_in", [])

            # Dataset evidence scores (replaces P_lit)
            enriched["P_evidence"]         = evidence_idx.get(pos, {}).get("P_evidence", 0.10)
            enriched["similar_examples"]   = evidence_idx.get(pos, {}).get("similar_examples", 0)
            enriched["methylated"]         = evidence_idx.get(pos, {}).get("methylated", 0)
            enriched["evidence_tier"]      = evidence_idx.get(pos, {}).get("evidence_tier", "none")
            enriched["top_similar_motifs"] = evidence_idx.get(pos, {}).get("top_similar_motifs", [])

            # Structural scores
            enriched["P_struct"]               = struct_idx.get(pos, {}).get("P_struct", 0.50)
            enriched["plddt"]                  = struct_idx.get(pos, {}).get("plddt", None)
            enriched["accessibility_category"] = struct_idx.get(pos, {}).get(
                "accessibility_category", "unknown (no accession)")

            merged.append(enriched)

        return merged

    # ── Main Pipeline ─────────────────────────────────────────────────────────

    def execute_analysis(self, query: str, accession: str = "") -> dict:
        """
        Full dataset-centric multi-agent prediction pipeline.

        Args:
            query: User query string. Accepted formats:
                   • "Analyze motif: FTLQIRGRERF"
                   • "Analyze sequence: MKRGGKPLRTA..."
                   • "Analyze sample #125 from dataset"
                   • "Analyze P04637 for methylation"
                   • Bare amino acid sequence
            accession: Optional UniProt accession (for structural agent only)

        Returns:
            Complete structured result dict with all agent scores,
            final predictions, and explanation data.
        """
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Query: '{query}'")
        print(f"{'='*60}")

        # ── STEP 1: Parse query ────────────────────────────────────────────────
        parsed = self.parse_query(query)
        if "error" in parsed:
            return {"success": False, "error": parsed["error"]}

        query_id = parsed["query_id"]
        mode     = parsed["mode"]
        if mode == "accession":
            accession = parsed["accession"]
        print(f"[Orchestrator] STEP 1 ✓  Mode: {mode}  | Query ID: {query_id}")

        # ── STEP 2: Resolve sequence ───────────────────────────────────────────
        sequence = self._resolve_sequence(parsed)
        if not sequence:
            return {
                "success": False,
                "error":   f"Could not resolve a sequence from query (mode={mode})."
            }
        print(f"[Orchestrator] STEP 2 ✓  Sequence length: {len(sequence)} aa")

        # ── STEP 3: HybridARGME prediction ────────────────────────────────────
        print(f"\n[Orchestrator] STEP 3: HybridARGME Agent...")
        pred_result = self.hybridargme_agent.execute({"sequence": sequence})
        sites       = pred_result.get("predictions", [])
        print(f"[Orchestrator] STEP 3 ✓  Scored {len(sites)} candidate sites.")

        if not sites:
            return {
                "success": False,
                "error":   "No R/K residues found in sequence."
            }

        # ── STEP 4: Parallel agent scoring ────────────────────────────────────
        print(f"\n[Orchestrator] STEP 4: Parallel Agent Scoring...")
        sites = self._run_parallel_scoring(sites, accession=accession)
        print(f"[Orchestrator] STEP 4 ✓")

        # ── STEP 5: Consensus Fusion ───────────────────────────────────────────
        print(f"\n[Orchestrator] STEP 5: Consensus Fusion Agent...")
        final_result      = self.consensus_agent.execute({"sites": sites})
        final_predictions = final_result.get("final_predictions", sites)

        positive_sites = [s for s in final_predictions if s.get("label", 0) == 1]
        positive_sites = sorted(positive_sites,
                                key=lambda x: x.get("P_final", 0), reverse=True)
        print(f"[Orchestrator] STEP 5 ✓  Predicted methylated sites: {len(positive_sites)}")

        # ── STEP 6: Explainer (top 3 positive sites) ──────────────────────────
        print(f"\n[Orchestrator] STEP 6: Explainer Agent...")
        targets_to_explain = positive_sites[:3]
        if not targets_to_explain and final_predictions:
            targets_to_explain = [
                sorted(final_predictions,
                       key=lambda x: x.get("P_final", 0), reverse=True)[0]
            ]

        explanation_reports = []
        for site in targets_to_explain:
            exp_data = self.explainer_agent.execute({"site_info": site})
            explanation_reports.append(exp_data)

        print(f"[Orchestrator] STEP 6 ✓  Explained {len(explanation_reports)} sites.")

        # ── Assemble final payload ─────────────────────────────────────────────
        print(f"\n[Orchestrator] Assembling final report payload...")

        fusion_mode = (final_predictions[0].get("fusion_mode", "N/A")
                       if final_predictions else "N/A")

        analysis_summary = {
            "success":        True,
            "query":          query,
            "query_id":       query_id,
            "query_mode":     mode,
            "ptm_type":       "methyl",
            "input_sequence": sequence,
            "predictions_summary": {
                "total_sites_checked":  pred_result.get("total_sites_checked"),
                "positive_sites_count": len(positive_sites),
                "threshold_used":       self.consensus_agent.threshold,
                "fusion_mode":          fusion_mode,
                "consensus_weights":    self.consensus_agent.weights,
            },
            "all_predictions":      final_predictions,
            "positive_predictions": positive_sites,
            "explanations":         explanation_reports,
            "metadata": {
                "protein_name": query_id,
                "gene_name":    mode,
                "organism":     "N/A",
                "function":     f"Dataset-centric methylation analysis. Query: {query}",
            },
            "uniprot_id":         accession or "N/A",
            "pubmed_literature":  [],
        }

        print(f"\n[Orchestrator] ✅ Analysis complete. Query ID: {query_id}.")
        return analysis_summary


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    orch = Orchestrator()

    test_queries = [
        "Analyze sequence: MKRGGKPLRTAMKRGGKPLRTA",
        "Analyze motif: XXXRGGRVLSX",
        "Analyze sample #0 from dataset",
    ]

    for q in test_queries:
        res = orch.execute_analysis(q)
        if res["success"]:
            print(f"\nQuery '{q}'")
            print(f"  Positive sites: {len(res.get('positive_predictions', []))}")
            for site in res.get("positive_predictions", []):
                print(
                    f"  {site['residue']}{site['position']}  "
                    f"P_final={site['P_final']:.3f}  [{site['confidence']}]  "
                    f"Hybrid={site['P_hybrid']:.2f} "
                    f"Motif={site['P_motif']:.2f} "
                    f"Evidence={site['P_evidence']:.2f} "
                    f"Conserv={site['P_conserv']:.2f}"
                )
        else:
            print(f"\n[FAIL] {q}: {res['error']}")

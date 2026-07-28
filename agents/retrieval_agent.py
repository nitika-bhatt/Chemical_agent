"""
Retrieval Agent
===============
Fetches protein metadata from UniProt REST API and literature from PubMed.

Extended in v2:
  - Now also extracts KNOWN methylation site positions from UniProt annotations.
    These positions are used by the Decision Agent as P_lit evidence.
"""

import requests
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote


class RetrievalAgent:
    def __init__(self):
        self.uniprot_url      = "https://rest.uniprot.org/uniprotkb"
        self.pubmed_search_url  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.pubmed_summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    # ── UniProt ───────────────────────────────────────────────────────────────

    def fetch_uniprot_data(self, accession: str) -> dict:
        """
        Fetches sequence, annotations, and known PTM positions from UniProt.

        Returns dict with:
          sequence, protein_name, gene_name, organism, function,
          known_ptms (list of {type, position}),
          known_methylation_positions (list of int) ← NEW
        """
        accession = accession.strip().upper()
        url = f"{self.uniprot_url}/{accession}.json"

        print(f"[Retrieval Agent] Querying UniProt for {accession}...")
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                print(f"[Retrieval Agent] UniProt returned {response.status_code}.")
                return {}

            data = response.json()

            # ── Basic fields ──────────────────────────────────────────────────
            sequence = data.get("sequence", {}).get("value", "")

            protein_names = data.get("proteinDescription", {}).get("recommendedName", {})
            protein_name  = protein_names.get("fullName", {}).get("value", "Unknown Protein")

            genes     = data.get("genes", [{}])
            gene_name = (genes[0].get("geneName", {}).get("value", "Unknown Gene")
                         if genes else "Unknown Gene")

            organism  = data.get("organism", {}).get("scientificName", "Unknown Organism")

            # ── Function ──────────────────────────────────────────────────────
            comments = data.get("comments", [])
            function_desc = "No functional description available."
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_desc = texts[0].get("value", "")
                        break

            # ── PTM features ──────────────────────────────────────────────────
            features  = data.get("features", [])
            known_ptms = []
            known_methylation_positions = []   # ← NEW: used for P_lit scoring

            for feat in features:
                if feat.get("type") == "Modified residue":
                    description = feat.get("description", "")
                    location    = feat.get("location", {})
                    pos         = location.get("start", {}).get("value", None)
                    if pos:
                        known_ptms.append({"type": description, "position": pos})

                        # NEW: collect positions where methylation is annotated
                        if "methyl" in description.lower():
                            known_methylation_positions.append(int(pos))

            return {
                "accession":                   accession,
                "protein_name":                protein_name,
                "gene_name":                   gene_name,
                "organism":                    organism,
                "sequence":                    sequence,
                "function":                    function_desc,
                "known_ptms":                  known_ptms,
                "known_methylation_positions": known_methylation_positions,  # ← NEW
            }

        except Exception as e:
            print(f"[Retrieval Agent] Error querying UniProt: {e}")
            return {}

    # ── PubMed ────────────────────────────────────────────────────────────────

    def search_pubmed_literature(self, protein_name: str, gene_name: str,
                                  ptm_type: str = "methylation") -> list:
        """Queries PubMed for literature linking protein to the PTM type."""
        query_term = f'("{protein_name}" OR "{gene_name}") AND {ptm_type}'
        print(f"[Retrieval Agent] PubMed query: {query_term}")

        search_params = {
            "db":      "pubmed",
            "term":    query_term,
            "retmode": "json",
            "retmax":  5,
        }

        articles = []
        try:
            res_search = requests.get(self.pubmed_search_url,
                                       params=search_params, timeout=15)
            if res_search.status_code != 200:
                return []

            id_list = (res_search.json()
                       .get("esearchresult", {}).get("idlist", []))
            if not id_list:
                print("[Retrieval Agent] No PubMed articles found.")
                return []

            summary_params = {
                "db":      "pubmed",
                "id":      ",".join(id_list),
                "retmode": "json",
            }
            res_summary = requests.get(self.pubmed_summary_url,
                                        params=summary_params, timeout=15)
            if res_summary.status_code != 200:
                return []

            results = res_summary.json().get("result", {})
            for uid in id_list:
                art   = results.get(uid, {})
                auths = art.get("authors", [])
                author_str = ", ".join(a.get("name", "") for a in auths[:3])
                if len(auths) > 3:
                    author_str += " et al."

                articles.append({
                    "pmid":    uid,
                    "title":   art.get("title", "No Title"),
                    "authors": author_str,
                    "journal": art.get("source", "Unknown Journal"),
                    "year":    art.get("pubdate", "").split(" ")[0],
                    "url":     f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                })

            print(f"[Retrieval Agent] Found {len(articles)} articles.")
            return articles

        except Exception as e:
            print(f"[Retrieval Agent] PubMed error: {e}")
            return []

    # ── P_lit scorer (NEW) ────────────────────────────────────────────────────

    def compute_plit_scores(self, site_list: list,
                             known_positions: list) -> list:
        """
        NEW: Assign a literature evidence score P_lit to each candidate site.

        P_lit = 1.0 if the exact position is annotated as methylated in UniProt.
        P_lit = 0.5 if an adjacent position (±2) is annotated.
        P_lit = 0.0 otherwise.

        Args:
            site_list:       List of site dicts with "position" key.
            known_positions: List of int positions from UniProt methylation annotations.

        Returns:
            Enriched site list with P_lit added.
        """
        known_set    = set(known_positions)
        adjacent_set = set()
        for p in known_positions:
            for delta in range(-2, 3):
                adjacent_set.add(p + delta)

        results = []
        for site in site_list:
            pos = site.get("position", -1)
            if pos in known_set:
                p_lit = 1.0
            elif pos in adjacent_set:
                p_lit = 0.5
            else:
                p_lit = 0.0
            enriched = dict(site)
            enriched["P_lit"] = p_lit
            results.append(enriched)

        return results

    # ── Main Execute ──────────────────────────────────────────────────────────

    def execute(self, query: str) -> dict:
        """Main execution: UniProt fetch + PubMed search."""
        accession = query.strip()
        if not accession:
            return {"error": "Empty query."}

        uniprot_info = self.fetch_uniprot_data(accession)
        if not uniprot_info:
            return {"error": f"Failed to retrieve data for {accession}."}

        lit = self.search_pubmed_literature(
            uniprot_info["protein_name"],
            uniprot_info["gene_name"],
            ptm_type="methylation"
        )
        uniprot_info["pubmed_articles"] = lit
        return uniprot_info


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = RetrievalAgent()
    res = agent.execute("P04637")
    print(json.dumps(res, indent=2))

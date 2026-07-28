"""
Structural Context Agent
========================
Evaluates whether a candidate methylation site is structurally accessible
(i.e., surface-exposed) using pLDDT-based heuristics.

In dataset-centric mode (no UniProt accession provided):
  - Returns a deterministic neutral P_struct = 0.50 for all sites.
  - This is the expected behaviour when running from a raw sequence or motif.

In accession mode (optional, if a UniProt accession is explicitly provided):
  - Queries AlphaFold EBI API for per-residue pLDDT scores.
  - Converts pLDDT to P_struct using the biological accessibility mapping.

pLDDT scoring (when available):
  > 90   : Very high confidence (rigid, buried → less likely accessible)
  70–90  : Confident (surface helix/strand)
  50–70  : Low confidence (loop → HIGH accessibility, good methylation target)
  < 50   : Very low confidence (disordered → can be methylated)

Reference:
  Jumper et al., Nature (2021) - AlphaFold2
  Hornbeck et al., Nucleic Acids Res. (2015) - PhosphoSitePlus accessibility
"""

import requests
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHAFOLD_API   = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
NEUTRAL_P_STRUCT = 0.50   # returned deterministically when no structural data


def plddt_to_accessibility(plddt: float) -> float:
    """
    Convert pLDDT score to a structural accessibility score.

    Biology:
      - pLDDT < 50  : disordered, very accessible → high score
      - pLDDT 50–70 : loop/coil region, accessible → highest score
      - pLDDT 70–85 : structured, moderate accessibility
      - pLDDT > 85  : rigid core, buried → low score
    """
    if plddt < 50:
        return 0.80   # disordered — high accessibility
    elif plddt < 70:
        return 0.95   # loop/coil sweet spot — best methylation target
    elif plddt < 85:
        return 0.60   # structured surface
    else:
        return 0.30   # rigid buried core


# ─────────────────────────────────────────────────────────────────────────────
# StructuralAgent Class
# ─────────────────────────────────────────────────────────────────────────────

class StructuralAgent:
    """
    Assigns structural accessibility scores (P_struct) to candidate sites.

    Primary mode (dataset-centric): No accession → deterministic neutral 0.50.
    Optional mode: If a UniProt accession is explicitly supplied, queries
                   AlphaFold EBI for per-residue pLDDT values.
    """

    def __init__(self):
        self.plddt_cache = {}   # {accession: [plddt_per_residue]}
        print("[Structural Agent] Initialized. "
              f"Neutral P_struct={NEUTRAL_P_STRUCT} when no accession given.")

    # ── API Fetch (optional, only when accession is supplied) ─────────────────

    def fetch_plddt_scores(self, accession: str) -> list:
        """
        Fetches per-residue pLDDT scores from AlphaFold EBI API.
        Only called when an explicit accession is provided.

        Returns:
            List of pLDDT floats (0-indexed). Empty list on failure.
        """
        accession = accession.strip().upper()

        if accession in self.plddt_cache:
            print(f"[Structural Agent] Using cached pLDDT for {accession}.")
            return self.plddt_cache[accession]

        print(f"[Structural Agent] Querying AlphaFold API for {accession}...")
        url = ALPHAFOLD_API.format(accession=accession)
        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                print(f"[Structural Agent] AlphaFold API returned "
                      f"{response.status_code} for {accession}.")
                return []

            data = response.json()
            if not data:
                return []

            entry      = data[0]
            plddt_list = (entry.get("plddt") or
                          [r.get("pLDDT", 0) for r in entry.get("residueScores", [])])
            self.plddt_cache[accession] = plddt_list
            print(f"[Structural Agent] Retrieved pLDDT for {len(plddt_list)} residues.")
            return plddt_list

        except Exception as e:
            print(f"[Structural Agent] Error fetching AlphaFold data: {e}")
            return []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_site(self, position_1based: int, plddt_scores: list) -> dict:
        """
        Compute P_struct for a single site.

        Args:
            position_1based: 1-based residue position.
            plddt_scores:    Full per-residue pLDDT list (may be empty).

        Returns:
            {"P_struct": float, "plddt": float|None, "accessibility_category": str}
        """
        idx = position_1based - 1   # convert to 0-based

        if not plddt_scores or idx < 0 or idx >= len(plddt_scores):
            # FIXED: return deterministic NEUTRAL_P_STRUCT (not random Gaussian)
            return {
                "P_struct":               NEUTRAL_P_STRUCT,
                "plddt":                  None,
                "accessibility_category": "unknown (no structural data)",
            }

        plddt    = float(plddt_scores[idx])
        p_struct = plddt_to_accessibility(plddt)

        if plddt < 50:
            category = "disordered (highly accessible)"
        elif plddt < 70:
            category = "loop/coil (accessible)"
        elif plddt < 85:
            category = "structured surface"
        else:
            category = "rigid core (buried)"

        return {
            "P_struct":               p_struct,
            "plddt":                  plddt,
            "accessibility_category": category,
        }

    def score_all_sites(self, site_list: list, accession: str = "") -> list:
        """
        Score all candidate sites.

        Args:
            site_list: List of site dicts with {"position": int, ...}
            accession: UniProt accession (optional). Empty string = neutral mode.

        Returns:
            Enriched site list with P_struct, plddt, accessibility_category added.
        """
        accession    = (accession or "").strip()
        plddt_scores = []

        if accession and accession not in ("N/A", "Unknown"):
            plddt_scores = self.fetch_plddt_scores(accession)
        else:
            print("[Structural Agent] No accession — "
                  f"returning deterministic neutral P_struct={NEUTRAL_P_STRUCT} for all sites.")

        results = []
        for site in site_list:
            pos          = site.get("position", 0)
            struct_res   = self.score_site(pos, plddt_scores)
            enriched     = dict(site)
            enriched["P_struct"]               = struct_res["P_struct"]
            enriched["plddt"]                  = struct_res["plddt"]
            enriched["accessibility_category"] = struct_res["accessibility_category"]
            results.append(enriched)

        scored = [r for r in results if r["plddt"] is not None]
        if scored:
            avg = np.mean([r["P_struct"] for r in scored])
            print(f"[Structural Agent] Scored {len(results)} sites. "
                  f"Avg P_struct = {avg:.3f}")
        else:
            print(f"[Structural Agent] Structural data unavailable. "
                  f"Using deterministic P_struct={NEUTRAL_P_STRUCT} for all sites.")

        return results

    # ── Orchestrator Interface ─────────────────────────────────────────────────

    def execute(self, data: dict) -> dict:
        """
        Orchestrator-compatible interface.

        Args:
            data: {"sites": list, "accession": str (optional)}
        Returns:
            {"scored_sites": list}
        """
        sites     = data.get("sites", [])
        accession = data.get("accession", "")

        if not sites:
            return {"error": "No sites provided to StructuralAgent.",
                    "scored_sites": []}

        scored = self.score_all_sites(sites, accession)
        return {"scored_sites": scored}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = StructuralAgent()

    test_sites = [
        {"position": 34,  "residue": "R", "motif": "XXXRGGRVLSX"},
        {"position": 89,  "residue": "K", "motif": "TGKGRGTAAXX"},
        {"position": 120, "residue": "R", "motif": "MAALEDRVVTX"},
    ]

    # Dataset-centric mode (no accession)
    result = agent.execute({"sites": test_sites})
    for s in result["scored_sites"]:
        print(
            f"  Pos {s['position']} {s['residue']}: "
            f"P_struct={s['P_struct']:.3f}  pLDDT={s['plddt']}  "
            f"[{s['accessibility_category']}]"
        )

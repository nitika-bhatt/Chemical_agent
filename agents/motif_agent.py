"""
Motif Context Agent
===================
Scans 11-mer sequence windows centered at candidate R/K sites for known
arginine and lysine methylation motifs. Returns a quantitative motif
confidence score P_motif in [0, 1].

Biological basis:
  - RGG / RXXG / GRG / RGRG : Arginine-Glycine-rich motifs (PRMT1 substrates)
  - GAR (Glycine Arginine Rich): Found in RBPs, ribosomal proteins
  - PGM : Proline-Glycine-Methionine motif
  - KXK / KXXK              : Lysine methylation context motifs (SET domain enzymes)
  - [RK]G[RK]               : General methyltransferase recognition pattern

Reference:
  Schapira et al., Nat. Rev. Mol. Cell Biol. (2011)
  Poulard et al., Mol. Cell (2017)
"""

import re
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Physicochemical Property Scales
# ─────────────────────────────────────────────────────────────────────────────
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2, 'X': 0.0
}


# ─────────────────────────────────────────────────────────────────────────────
# Motif Definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (pattern, weight, description)
#   Pattern: regex applied to the full 11-mer
#   Weight:  contribution to final score (0–1), biologically justified
METHYLATION_MOTIFS = [
    # Arginine-Glycine-Glycine (core PRMT1 motif)
    (re.compile(r"RGG"),          1.00, "RGG core motif (PRMT1)"),
    # Double RGG
    (re.compile(r"RGGR"),         1.00, "RGGR extended motif"),
    # Alternating RGRG
    (re.compile(r"RGRG"),         0.95, "RGRG repeat motif"),
    # GRG (glycine-flanked arginine)
    (re.compile(r"GRG"),          0.90, "GRG flanked arginine"),
    # RXR (arginine-x-arginine)
    (re.compile(r"R.R"),          0.80, "RXR di-arginine motif"),
    # GAR (glycine-arginine rich)
    (re.compile(r"G[AR]{2,}"),    0.85, "GAR-rich region"),
    # RG repeat
    (re.compile(r"(RG){2,}"),     0.90, "RG repeat"),
    # Lysine methylation: KXK
    (re.compile(r"K.K"),          0.70, "KXK di-lysine context"),
    # Lysine methylation: KXXK
    (re.compile(r"K..K"),         0.65, "KXXK context"),
    # General methyltransferase recognition
    (re.compile(r"[RK]G[RK]"),    0.80, "[RK]G[RK] general"),
    # Arginine in acidic context (less likely, lower weight)
    (re.compile(r"[DE]R[DE]"),    0.30, "Acidic-flanked R (less favorable)"),
]

# Center position in the 11-mer
CENTER = 5


# ─────────────────────────────────────────────────────────────────────────────
# MotifAgent Class
# ─────────────────────────────────────────────────────────────────────────────

class MotifAgent:
    """
    Scores an 11-mer window for methylation-relevant sequence motifs.

    P_motif is computed as a weighted sum of motif matches normalised to [0, 1].
    """

    def __init__(self):
        self.motifs = METHYLATION_MOTIFS
        print("[Motif Agent] Initialized with", len(self.motifs), "methylation motif rules.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_center_residue(self, motif_seq: str) -> bool:
        """Verify center residue is R or K (methylation target)."""
        if len(motif_seq) < CENTER + 1:
            return False
        return motif_seq[CENTER] in ('R', 'K')

    def _scan_motifs(self, motif_seq: str) -> list:
        """
        Scan the 11-mer for all motif matches.

        Returns:
            List of (description, weight) for each matched motif.
        """
        hits = []
        for pattern, weight, desc in self.motifs:
            if pattern.search(motif_seq):
                hits.append((desc, weight))
        return hits

    def _compute_score(self, hits: list, center_is_target: bool) -> float:
        """
        Aggregate motif hit weights into a single P_motif score.

        Strategy:
          - Start from 0.
          - Add each motif weight, clamped so total <= 1.
          - Penalise sites where center residue is not R/K.
          - Apply a base floor of 0.05 (even no-motif sites have some chance).
        """
        if not hits:
            score = 0.05  # minimal floor
        else:
            # Weighted sum, clipped to 1.0
            raw = sum(w for _, w in hits)
            # Normalise: more than one hit saturates score faster
            score = 1.0 - np.exp(-raw)   # exponential saturation

        if not center_is_target:
            score *= 0.3   # heavy penalty: non-R/K center

        return float(np.clip(score, 0.0, 1.0))

    def _compute_physical_properties(self, motif_seq: str) -> dict:
        """Calculate basic physicochemical properties of the 11-mer window."""
        seq = motif_seq.replace("X", "")
        if not seq:
            return {"hydrophobicity": 0.0, "net_charge": 0, "molecular_weight": 0.0}
        
        # 1. Average Hydrophobicity (Kyte-Doolittle scale)
        hydro = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / len(seq)
        
        # 2. Net Charge (approximate at pH 7)
        pos = sum(1 for aa in seq if aa in ['R', 'K'])
        neg = sum(1 for aa in seq if aa in ['D', 'E'])
        charge = pos - neg
        
        # 3. Approximate Molecular Weight (Daltons)
        mw = len(seq) * 110.0
        
        return {
            "hydrophobicity": round(hydro, 3),
            "net_charge": charge,
            "molecular_weight": mw
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def score_site(self, motif_seq: str) -> dict:
        """
        Score a single 11-mer window.

        Args:
            motif_seq: 11-character amino acid string (padded with 'X' at boundaries)

        Returns:
            {
              "P_motif": float,        # score in [0, 1]
              "matched_motifs": list,  # names of matched motifs
              "center_residue": str
            }
        """
        motif_seq = motif_seq.upper().strip()

        center_residue = motif_seq[CENTER] if len(motif_seq) > CENTER else "?"
        center_is_target = center_residue in ('R', 'K')

        hits = self._scan_motifs(motif_seq)
        score = self._compute_score(hits, center_is_target)
        phys_props = self._compute_physical_properties(motif_seq)

        # 🎯 THE BIOPHYSICAL FILTER: Methylation enzymes prefer hydrophilic (water-loving), 
        # disordered regions. If the region is highly hydrophobic (buried inside the protein),
        # we penalize the motif score because the enzyme cannot physically reach it!
        if phys_props["hydrophobicity"] > 0.5:
            score = score * 0.7  # 30% penalty for being physically inaccessible
        elif phys_props["hydrophobicity"] < -1.0:
            # Highly hydrophilic regions (like RGG/GAR) get a slight structural boost
            score = min(1.0, score * 1.1)

        return {
            "P_motif": round(score, 3),
            "matched_motifs": [desc for desc, _ in hits],
            "center_residue": center_residue,
            "physical_properties": phys_props
        }

    def score_all_sites(self, site_list: list) -> list:
        """
        Score all candidate sites in the prediction payload.

        Args:
            site_list: List of dicts with at least {"motif": str, "position": int, "residue": str}

        Returns:
            Same list enriched with "P_motif" and "matched_motifs" fields.
        """
        results = []
        for site in site_list:
            motif_seq = site.get("motif", "")
            result = self.score_site(motif_seq)
            enriched = dict(site)
            enriched["P_motif"] = result["P_motif"]
            enriched["matched_motifs"] = result["matched_motifs"]
            enriched["physical_properties"] = result["physical_properties"]
            results.append(enriched)

        print(f"[Motif Agent] Scored {len(results)} sites. "
              f"Avg P_motif = {np.mean([r['P_motif'] for r in results]):.3f}")
        return results

    def execute(self, data: dict) -> dict:
        """
        Orchestrator-compatible interface.

        Args:
            data: {"sites": list_of_site_dicts}
        Returns:
            {"scored_sites": list}
        """
        sites = data.get("sites", [])
        if not sites:
            return {"error": "No sites provided to MotifAgent.", "scored_sites": []}

        scored = self.score_all_sites(sites)
        return {"scored_sites": scored}


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MotifAgent()

    test_sites = [
        {"position": 5,  "residue": "R", "motif": "XXXRGGRVLSX"},
        {"position": 10, "residue": "R", "motif": "TGRGRGTAAXX"},
        {"position": 15, "residue": "K", "motif": "XXXKVLSKKXX"},
        {"position": 20, "residue": "R", "motif": "MAALEDRVVTX"},
    ]

    result = agent.execute({"sites": test_sites})
    for s in result["scored_sites"]:
        print(f"  Pos {s['position']} {s['residue']}: P_motif={s['P_motif']:.3f}  Hits={s['matched_motifs']}")

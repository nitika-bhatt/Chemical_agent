"""
Dataset Evidence Agent — Motif-Level Evidence Scoring from Training Data
========================================================================
Searches the local dataset for 11-mer motifs similar (Hamming ≤ 2) to each
candidate site and computes a Laplace-smoothed methylation evidence score.

Replaces the external UniProt P_lit pathway. No network calls.
Fully dataset-centric and local.
"""

import os
import numpy as np
import pandas as pd

SPECIES_LIST      = ["human", "mouse", "rat", "chimpanzee"]
HAMMING_THRESHOLD = 2
SMOOTHING         = 2
EVIDENCE_FLOOR    = 0.10
MOTIF_LEN         = 11

# Local-first data directory candidates (tried in order)
_DEFAULT_DATA_DIRS = [
    "data-11",
    os.path.join(os.path.dirname(__file__), "..", "data-11"),
    "/content/drive/MyDrive/ProteinMind_Agent/data-11",
    "/kaggle/working/ProteinMind_Agent/data-11",
]


def _resolve_data_dir(data_dir: str = None) -> str:
    """Return the first existing data directory from candidates."""
    candidates = ([data_dir] if data_dir else []) + _DEFAULT_DATA_DIRS
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return data_dir or _DEFAULT_DATA_DIRS[0]


def hamming_distance(s1: str, s2: str) -> int:
    """Compute Hamming distance between two equal-length strings."""
    if len(s1) != len(s2):
        return max(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


class DatasetEvidenceAgent:
    """
    Searches local CSV datasets for 11-mers similar to each candidate site.

    Evidence score P_evidence (Laplace-smoothed):
        P_evidence = (weighted_positive_count + α/2) / (total_weight + α)

    Tiers:
        strong : P_evidence ≥ 0.70
        weak   : P_evidence > EVIDENCE_FLOOR
        none   : P_evidence == EVIDENCE_FLOOR (no matches)
    """

    def __init__(self, data_dir=None):
        self.data_dir = _resolve_data_dir(data_dir)
        self.dataset  = []          # list of {"motif": str, "label": int, "species": str}
        self._load_all_species()
        print(f"[DatasetEvidence Agent] Loaded {len(self.dataset):,} total entries "
              f"from: {os.path.abspath(self.data_dir)}")

    def _load_all_species(self):
        """Walk every species sub-folder and collect all 11-mer rows."""
        for species in SPECIES_LIST:
            species_dir = os.path.join(self.data_dir, species.capitalize())
            if not os.path.exists(species_dir):
                species_dir = os.path.join(self.data_dir, species)
            if not os.path.exists(species_dir):
                continue

            loaded = 0
            for root, _, files in os.walk(species_dir):
                for file in files:
                    if not file.endswith(".csv"):
                        continue
                    fpath = os.path.join(root, file)
                    try:
                        df = pd.read_csv(fpath)

                        # Infer label from filename if column missing
                        if "label" not in df.columns:
                            if "positive" in file.lower():
                                df["label"] = 1
                            elif "negative" in file.lower():
                                df["label"] = 0
                            else:
                                continue

                        df["label"] = pd.to_numeric(df["label"], errors="coerce")
                        df.dropna(subset=["label"], inplace=True)
                        df["label"] = df["label"].astype(int)

                        # Normalise sequence column name
                        # Real dataset uses 'Positive_sequence' or 'Negative_sequence'
                        if "sequence" not in df.columns:
                            for candidate in ["Positive_sequence", "Negative_sequence",
                                              "Sequence", "sequence"]:
                                if candidate in df.columns:
                                    df = df.rename(columns={candidate: "sequence"})
                                    break
                            else:
                                df = df.rename(columns={df.columns[-1]: "sequence"})

                        df["sequence"] = (df["sequence"]
                                          .astype(str)
                                          .str.upper()
                                          .str.strip())
                        # Keep only valid MOTIF_LEN 11-mers
                        df = df[df["sequence"].str.len() == MOTIF_LEN]

                        for _, row in df.iterrows():
                            self.dataset.append({
                                "motif":   row["sequence"],
                                "label":   int(row["label"]),
                                "species": species,
                            })
                            loaded += 1
                    except Exception:
                        pass

            if loaded:
                print(f"[DatasetEvidence Agent]   {species.capitalize():12s}: "
                      f"{loaded:,} entries loaded.")

    # ── Similarity search ──────────────────────────────────────────────────────

    def _find_similar(self, query_motif: str) -> list:
        """Find all dataset entries within Hamming distance threshold."""
        similar = []
        for entry in self.dataset:
            d = hamming_distance(query_motif, entry["motif"])
            if d == 0:
                similar.append({
                    **entry, "distance": d,
                    "weight": 1.0, "match_type": "exact"
                })
            elif d <= HAMMING_THRESHOLD:
                weight = 1.0 - (d / (HAMMING_THRESHOLD + 1))
                similar.append({
                    **entry, "distance": d,
                    "weight": weight, "match_type": "near"
                })
        return similar

    def _compute_evidence_score(self, similar: list) -> float:
        """Laplace-smoothed evidence score."""
        if not similar:
            return EVIDENCE_FLOOR
        total_w    = sum(e["weight"] for e in similar)
        positive_w = sum(e["weight"] for e in similar if e["label"] == 1)
        return float(np.clip(
            (positive_w + SMOOTHING / 2) / (total_w + SMOOTHING),
            EVIDENCE_FLOOR, 1.0
        ))

    # ── Scoring API ────────────────────────────────────────────────────────────

    def score_site(self, motif_seq: str) -> dict:
        """Score one 11-mer for dataset evidence."""
        motif_seq = motif_seq.upper().strip()
        similar   = self._find_similar(motif_seq)
        p_ev      = self._compute_evidence_score(similar)

        if p_ev >= 0.70:
            tier = "strong"
        elif p_ev > EVIDENCE_FLOOR:
            tier = "weak"
        else:
            tier = "none"

        return {
            "P_evidence":        p_ev,
            "similar_examples":  len(similar),
            "methylated":        sum(1 for e in similar if e["label"] == 1),
            "evidence_tier":     tier,
            "top_similar_motifs": sorted(similar, key=lambda x: x["distance"])[:3],
        }

    def score_all_sites(self, site_list: list) -> list:
        """Score all candidate sites."""
        return [{**site, **self.score_site(site.get("motif", ""))}
                for site in site_list]

    def execute(self, data: dict) -> dict:
        """Orchestrator-compatible interface."""
        return {"scored_sites": self.score_all_sites(data.get("sites", []))}

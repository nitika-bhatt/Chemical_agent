"""
Conservation Agent — Dataset-Centric Cross-Species Conservation Scoring
=======================================================================
Loads methylation site data from local dataset CSVs for Human, Mouse, Rat,
and Chimpanzee and assigns a conservation score P_conserv based on how many
species share an identical 11-mer motif.

No external API calls. Fully local.
"""

import os
import pandas as pd

SPECIES_LIST = ["human", "mouse", "rat", "chimpanzee"]

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


class ConservationAgent:
    """
    Assigns cross-species conservation scores (P_conserv) to candidate sites.

    P_conserv:
      ≥ 3 species sharing the motif → 0.90 (highly conserved)
      2 species                     → 0.60 (moderately conserved)
      1 species                     → 0.35 (weakly conserved)
      0 species                     → 0.10 (no conservation evidence)
    """

    def __init__(self, data_dir=None):
        self.data_dir     = _resolve_data_dir(data_dir)
        self.species_sites = {sp: set() for sp in SPECIES_LIST}
        self._load_species_data()

    def _load_species_data(self):
        """Load all positive methylation 11-mers from each species CSV."""
        for species in SPECIES_LIST:
            # Try capitalised then lowercase folder name
            species_dir = os.path.join(self.data_dir, species.capitalize())
            if not os.path.exists(species_dir):
                species_dir = os.path.join(self.data_dir, species)
            if not os.path.exists(species_dir):
                continue

            positives = []
            for root, _, files in os.walk(species_dir):
                for file in files:
                    if not file.endswith(".csv"):
                        continue
                    try:
                        df = pd.read_csv(os.path.join(root, file))

                        # Infer label from filename if column missing
                        if "label" not in df.columns:
                            if "positive" in file.lower():
                                df["label"] = 1
                            elif "negative" in file.lower():
                                df["label"] = 0
                            else:
                                continue

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

                        if "sequence" in df.columns:
                            # If label column doesn't exist, infer from filename
                            if "label" not in df.columns:
                                if "positive" in file.lower():
                                    df["label"] = 1
                                elif "negative" in file.lower():
                                    df["label"] = 0

                        if "label" in df.columns and "sequence" in df.columns:
                            pos_seqs = (df[df["label"] == 1]["sequence"]
                                        .astype(str)
                                        .str.upper()
                                        .str.strip()
                                        .tolist())
                            positives.extend(pos_seqs)
                    except Exception:
                        pass

            if positives:
                self.species_sites[species] = set(positives)
                print(f"[Conservation Agent] Loaded {len(positives):,} "
                      f"positive sites for {species.capitalize()}.")

        total = sum(len(v) for v in self.species_sites.values())
        print(f"[Conservation Agent] Total conservation pool: {total:,} entries "
              f"across {len([s for s in SPECIES_LIST if self.species_sites[s]])} species.")

    def score_site(self, motif_seq: str) -> dict:
        """Score one 11-mer for cross-species conservation."""
        motif_seq    = motif_seq.upper().strip()
        conserved_in = [sp for sp, sites in self.species_sites.items()
                        if motif_seq in sites]
        c = len(conserved_in)
        if c >= 3:
            p_conserv = 0.90
        elif c == 2:
            p_conserv = 0.60
        elif c == 1:
            p_conserv = 0.35
        else:
            p_conserv = 0.10
        return {"P_conserv": p_conserv, "conserved_in": conserved_in}

    def score_all_sites(self, site_list: list) -> list:
        """Score all candidate sites for conservation."""
        return [{**site, **self.score_site(site.get("motif", ""))}
                for site in site_list]

    def execute(self, data: dict) -> dict:
        """Orchestrator-compatible interface."""
        return {"scored_sites": self.score_all_sites(data.get("sites", []))}

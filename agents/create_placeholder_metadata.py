import os

# Base data directory (relative to the repository root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data-11"))

species = ["Human", "Chimpanzee", "Mouse", "Rat"]
ptms = [
    "Asymmetric_dimethylarginine",
    "Dimethylated_arginine",
    "Omega-N-methylarginine",
    "Symmetric_dimethylarginine",
]
labels = ["positive", "negative"]

for sp in species:
    for ptm in ptms:
        folder = os.path.join(BASE_DIR, sp, ptm)
        os.makedirs(folder, exist_ok=True)
        for lbl in labels:
            filename = f"{sp.lower()}_{lbl}_{ptm}_metadata.csv"
            path = os.path.join(folder, filename)
            content = (
                "Protein Name,UniProt ID,Gene Symbol,Organism,Function\n"
                f"{sp} {ptm} {lbl} Protein,P00000,{sp}_GENE,{sp},Placeholder {ptm} {lbl}"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created: {path}")

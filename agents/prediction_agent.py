"""
HybridARGME Prediction Agent
=============================
Loads the trained 5-fold ensemble of HybridARGME models (stored as
DeepTESite architecture in model_core.py) and predicts methylation
probability P_hybrid for every R/K candidate site in a protein sequence.

This agent is Agent-1 in the parallel multi-agent pipeline.
Output P_hybrid feeds into the Decision Fusion Agent.
"""

import os
import torch
import numpy as np
from model_core import DeepTESite, SEQ_LEN, INPUT_DIM


class HybridARGMEAgent:
    """
    Agent 1: HybridARGME Deep Learning Predictor.
    Produces P_hybrid (transformer ensemble probability) for each site.
    Formerly called PredictionAgent — renamed to match the thesis model name.
    """

    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.avg_threshold = 0.5

        # Lazy-load ESM-2
        self.esm_model = None
        self.alphabet = None
        self.batch_converter = None

        # Find checkpoint
        if checkpoint_path is None:
            paths = [
                "/kaggle/working/checkpoints/pretrained_esm_v4.pt",
                "/content/checkpoints/pretrained_esm_v4.pt",
                "checkpoints/pretrained_esm_v4.pt",
                "../checkpoints/pretrained_esm_v4.pt",
            ]
            for p in paths:
                if os.path.exists(p):
                    checkpoint_path = p
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_ensemble(checkpoint_path)
        else:
            print("[HybridARGME Agent] WARNING: Checkpoint not found. "
                  "Running with untrained model weights.")
            self.models.append(DeepTESite().to(self.device))

    # ── Ensemble Loading ──────────────────────────────────────────────────────

    def load_ensemble(self, checkpoint_path: str):
        """Load all fold models from the saved ensemble checkpoint."""
        print(f"[HybridARGME Agent] Loading ensemble from {checkpoint_path}...")
        try:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
            fold_states = checkpoint.get("fold_states", [])
            self.avg_threshold = checkpoint.get("avg_threshold", 0.5)

            self.models = []
            for state in fold_states:
                model = DeepTESite().to(self.device)
                model.load_state_dict(state)
                model.eval()
                self.models.append(model)

            print(f"[HybridARGME Agent] Loaded {len(self.models)} fold models. "
                  f"Threshold={self.avg_threshold:.3f}")
        except Exception as e:
            print(f"[HybridARGME Agent] Error loading ensemble: {e}")
            self.models = [DeepTESite().to(self.device)]

    # ── ESM-2 Embedding ───────────────────────────────────────────────────────

    def _init_esm_model(self):
        """Lazy-load ESM-2 only when first needed."""
        if self.esm_model is None:
            print("[HybridARGME Agent] Loading ESM-2 model...")
            try:
                model, alphabet = torch.hub.load("facebookresearch/esm", "esm2_t6_8M_UR50D", trust_repo=True)
            except Exception as e:
                raise RuntimeError(f"[HybridARGME Agent] Failed to load ESM-2 from torch.hub: {e}\n""Ensure internet access and that torch.hub cache is writable.") from e
            self.esm_model = model.eval().to(self.device)
            self.alphabet = alphabet
            self.batch_converter = alphabet.get_batch_converter()
            print("[HybridARGME Agent] ESM-2 ready.")

    def get_esm_embedding(self, sequence: str) -> torch.Tensor:
        """Returns ESM-2 per-residue embeddings for a sequence. Shape: [L, 320]"""
        self._init_esm_model()
        data = [("seq_0", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[6])

        emb = results["representations"][6]
        return emb[0, 1:len(sequence) + 1, :].cpu()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_sites(self, sequence: str) -> list:
        """
        Predict methylation probability for every R/K site in the sequence.

        Returns:
            List of dicts with position, residue, motif, P_hybrid, prediction.
        """
        sequence = sequence.strip().upper()
        padded = "XXXXX" + sequence + "XXXXX"
        target_positions = [i for i, c in enumerate(sequence) if c in ('R', 'K')]

        if not target_positions:
            print("[HybridARGME Agent] No R/K residues found.")
            return []

        print(f"[HybridARGME Agent] Scoring {len(target_positions)} candidate sites...")

        results = []
        for pos in target_positions:
            motif_seq = padded[pos: pos + 11]

            try:
                emb = self.get_esm_embedding(motif_seq)   # [11, 320]
            except Exception as e:
                print(f"[HybridARGME Agent] ESM error at pos {pos}: {e}")
                emb = torch.zeros(11, INPUT_DIM)

            x = emb.unsqueeze(0).to(self.device)   # [1, 11, 320]

            probs = []
            for model in self.models:
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.softmax(logits, dim=1)[:, 1].item()
                    probs.append(prob)

            avg_prob = float(np.mean(probs))
            pred_class = 1 if avg_prob >= self.avg_threshold else 0

            results.append({
                "position":   pos + 1,          # 1-based
                "residue":    sequence[pos],
                "motif":      motif_seq,
                "P_hybrid":   avg_prob,
                "prediction": pred_class,
                "ptm_type":   "methyl",
            })

        return results

    # ── Orchestrator Interface ─────────────────────────────────────────────────

    def execute(self, data: dict) -> dict:
        """Orchestrator-compatible interface."""
        sequence = data.get("sequence", "")
        if not sequence:
            return {"error": "No sequence provided to HybridARGMEAgent."}

        predictions = self.predict_sites(sequence)
        positives = [p for p in predictions if p["prediction"] == 1]

        return {
            "ptm_type":           "methyl",
            "total_sites_checked": len(predictions),
            "positive_sites_count": len(positives),
            "predictions":         predictions,
        }


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Keeps old code working if anything still imports PredictionAgent
PredictionAgent = HybridARGMEAgent


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = HybridARGMEAgent()
    test_seq = "MQAALRKVLSRGGRVVTX"
    res = agent.execute({"sequence": test_seq})
    import json
    print(json.dumps(res, indent=2))

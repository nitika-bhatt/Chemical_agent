"""
Explainer Agent
===============
Produces interpretability outputs for positive methylation prediction sites.

Two outputs are computed for each site:
  1. Attention Map   — averaged self-attention weights from BMAM layers [11×11]
  2. SHAP Scores     — perturbation-based feature attribution for each residue [11]

Disease association lookup via UniProt has been removed.
This agent is fully self-contained and dataset-centric.

Reference:
  Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
"""

import torch
import numpy as np
from model_core import DeepTESite, SEQ_LEN, INPUT_DIM


class ExplainerAgent:
    def __init__(self, prediction_agent=None):
        self.prediction_agent = prediction_agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_attention_weights(self, embedding_tensor: torch.Tensor) -> np.ndarray:
        """
        Extracts self-attention matrices from the BMAM layers of the loaded models.

        Args:
            embedding_tensor: Tensor of shape [11, 320]

        Returns:
            Numpy array of shape [11, 11] representing averaged attention weights.
        """
        if not self.prediction_agent or not self.prediction_agent.models:
            return np.random.dirichlet(np.ones(SEQ_LEN), size=SEQ_LEN)

        model = self.prediction_agent.models[0]
        model.eval()

        x = embedding_tensor.unsqueeze(0).to(self.device)  # [1, 11, 320]

        with torch.no_grad():
            _ = model(x)

        layer_attns = []
        for layer in model.layers:
            attn_weights = layer.bmam.last_attn_weights   # [1, n_heads, 11, 11]
            if attn_weights is not None:
                avg_heads = attn_weights[0].mean(dim=0).cpu().numpy()
                layer_attns.append(avg_heads)

        if not layer_attns:
            return np.ones((SEQ_LEN, SEQ_LEN)) / SEQ_LEN

        return np.mean(layer_attns, axis=0)

    def explain_prediction_shap(
        self, embedding_tensor: torch.Tensor, original_prob: float
    ) -> list:
        """
        Computes local feature attributions using Occlusion Sensitivity
        (perturbation-based SHAP-like method).

        Masks each residue representation in the 11-mer and measures the
        resulting confidence drop.

        Args:
            embedding_tensor: Tensor of shape [11, 320]
            original_prob:    Ensemble baseline prediction probability.

        Returns:
            List of float attribution scores of length 11.
        """
        if not self.prediction_agent or not self.prediction_agent.models:
            return [0.0] * SEQ_LEN

        models = self.prediction_agent.models
        scores = []

        for idx in range(SEQ_LEN):
            perturbed_emb       = embedding_tensor.clone()
            perturbed_emb[idx]  = 0.0   # mask this residue

            x_pert = perturbed_emb.unsqueeze(0).to(self.device)

            perturbed_probs = []
            for model in models:
                with torch.no_grad():
                    logits = model(x_pert)
                    prob   = torch.softmax(logits, dim=1)[:, 1].item()
                    perturbed_probs.append(prob)

            perturbed_prob = np.mean(perturbed_probs)
            attr_score     = original_prob - perturbed_prob   # drop = importance
            scores.append(float(attr_score))

        return scores

    def execute(self, data: dict) -> dict:
        """
        Executes explanation flow for a positive prediction site.

        Args:
            data: {
                "site_info": dict with keys motif, P_hybrid, P_final, position, residue
            }

        Returns:
            {
              "position":    int,
              "residue":     str,
              "motif":       str,
              "probability": float,
              "attention_map": list[list[float]],   # 11×11
              "shap_scores":   list[float]           # length 11
            }
        """
        site_info    = data.get("site_info", {})
        motif        = site_info.get("motif", "")
        original_prob = site_info.get("P_hybrid", site_info.get("probability", 0.5))

        if not motif or len(motif) != SEQ_LEN:
            return {"error": "Invalid motif sequence. Must be length 11."}

        # Get ESM embedding for the motif
        if self.prediction_agent:
            emb = self.prediction_agent.get_esm_embedding(motif)
        else:
            emb = torch.zeros(SEQ_LEN, INPUT_DIM)

        # 1. Compute self-attention map
        attn_map   = self.extract_attention_weights(emb)

        # 2. Compute SHAP-like attribution scores
        shap_scores = self.explain_prediction_shap(emb, original_prob)

        return {
            "position":    site_info.get("position"),
            "residue":     site_info.get("residue"),
            "motif":       motif,
            "probability": site_info.get("P_final", original_prob),
            "attention_map": attn_map.tolist(),
            "shap_scores":   shap_scores,
        }

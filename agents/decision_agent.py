"""
Decision Fusion Agent  ─  OOF Meta-Learner Training Helper
==========================================================
This module is kept for backward-compatibility and OOF meta-learner training.
At inference time, ConsensusAgent (consensus_agent.py) is used instead.

Agent scores fused (dataset-centric — no UniProt):
  • P_hybrid   – HybridARGME transformer probability
  • P_motif    – Sequence motif match score
  • P_struct   – AlphaFold structural accessibility (neutral 0.50 without accession)
  • P_conserv  – Cross-species conservation score (dataset-based)
  • P_evidence – Dataset evidence score (replaces former P_lit from UniProt)

The meta-learner is trained using out-of-fold (OOF) predictions from the
same 5-fold cross-validation used for HybridARGME, so no data leakage occurs.

If no trained meta-learner is found, a biologically-motivated weighted
average is used as the default fusion rule.

Reference:
  Wolpert (1992) "Stacked Generalization" – Neural Networks
  Zhou (2012) "Ensemble Methods" – CRC Press
"""

import os
import json
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (matthews_corrcoef, f1_score,
                                  roc_auc_score, accuracy_score,
                                  precision_score, recall_score)
    import joblib
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[Decision Agent] sklearn not found — using weighted average fallback.")


# ─────────────────────────────────────────────────────────────────────────────
# Default weights (dataset-centric — P_lit replaced by P_evidence)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "P_hybrid":   0.45,   # Deep learning — primary signal
    "P_motif":    0.20,   # Sequence motif — strong biological prior
    "P_evidence": 0.20,   # Dataset evidence (replaces P_lit from UniProt)
    "P_conserv":  0.10,   # Cross-species conservation
    "P_struct":   0.05,   # Structure accessibility (neutral without accession)
}

SCORE_KEYS = ["P_hybrid", "P_motif", "P_evidence", "P_conserv", "P_struct"]

DEFAULT_MODEL_PATH = "checkpoints/decision_agent_meta_learner.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# DecisionAgent Class
# ─────────────────────────────────────────────────────────────────────────────

class DecisionAgent:
    """
    Fuses 5 agent evidence scores into a final methylation prediction.

    Two modes:
      1. Meta-Learner (trained): Logistic Regression trained on OOF predictions.
      2. Weighted Average (fallback): Uses DEFAULT_WEIGHTS.
    """

    def __init__(self, model_path: str = None, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.weights = DEFAULT_WEIGHTS.copy()

        # Try to load a saved meta-learner
        mp = model_path or DEFAULT_MODEL_PATH
        if os.path.exists(mp):
            self._load_model(mp)
        else:
            print("[Decision Agent] No trained meta-learner found. "
                  "Using biologically-weighted average as default fusion rule.")

    # ── Model I/O ─────────────────────────────────────────────────────────────

    def _load_model(self, path: str):
        if not _SKLEARN_OK:
            return
        try:
            bundle = joblib.load(path)
            self.model   = bundle["model"]
            self.scaler  = bundle["scaler"]
            self.threshold = bundle.get("threshold", 0.5)
            self.is_trained = True
            print(f"[Decision Agent] Loaded trained meta-learner from {path}. "
                  f"Threshold={self.threshold:.3f}")
        except Exception as e:
            print(f"[Decision Agent] Could not load model: {e}")

    def save_model(self, path: str = DEFAULT_MODEL_PATH):
        if not (self.is_trained and _SKLEARN_OK):
            print("[Decision Agent] No model to save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model,
                     "scaler": self.scaler,
                     "threshold": self.threshold}, path)
        print(f"[Decision Agent] Meta-learner saved to {path}.")

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Logistic Regression meta-learner.

        Args:
            X: shape (N, 5) — columns = SCORE_KEYS order
                              [P_hybrid, P_motif, P_evidence, P_conserv, P_struct]
            y: shape (N,)   — binary labels {0, 1}

        Call this after collecting OOF (out-of-fold) predictions from all agents
        on the validation set of each fold.
        """
        if not _SKLEARN_OK:
            print("[Decision Agent] sklearn not available — cannot train meta-learner.")
            return

        print(f"[Decision Agent] Training meta-learner on {X.shape[0]} samples...")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        base_lr = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42, solver="lbfgs"
        )
        # Calibrate probabilities with Platt scaling
        self.model = CalibratedClassifierCV(base_lr, cv=3, method="sigmoid")
        self.model.fit(X_scaled, y)

        # Find optimal threshold on training data using MCC
        probs_train = self.model.predict_proba(X_scaled)[:, 1]
        best_mcc, best_thr = -1.0, 0.5
        for thr in np.arange(0.2, 0.8, 0.01):
            preds = (probs_train >= thr).astype(int)
            mcc = matthews_corrcoef(y, preds)
            if mcc > best_mcc:
                best_mcc, best_thr = mcc, thr
        self.threshold = best_thr
        self.is_trained = True

        # Report training performance
        final_preds = (probs_train >= self.threshold).astype(int)
        print(f"[Decision Agent] Training complete.")
        print(f"  Optimal threshold : {self.threshold:.3f}")
        print(f"  MCC               : {matthews_corrcoef(y, final_preds):.4f}")
        print(f"  F1                : {f1_score(y, final_preds):.4f}")
        print(f"  AUC               : {roc_auc_score(y, probs_train):.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _weighted_average(self, scores: dict) -> float:
        """Fallback: biologically motivated weighted average."""
        total_w = sum(self.weights.values())
        p_final = sum(scores.get(k, 0.5) * w
                      for k, w in self.weights.items()) / total_w
        return float(np.clip(p_final, 0.0, 1.0))

    def fuse(self, scores: dict) -> dict:
        """
        Fuse 5 agent scores into a final prediction for ONE site.

        Args:
            scores: dict with keys P_hybrid, P_motif, P_evidence, P_conserv, P_struct

        Returns:
            {
              "P_final"    : float,   # fused probability
              "label"      : int,     # 0 or 1
              "confidence" : str,     # human-readable confidence tier
              "fusion_mode": str      # "meta_learner" or "weighted_average"
            }
        """
        if self.is_trained and _SKLEARN_OK:
            x_vec = np.array([[scores.get(k, 0.5) for k in SCORE_KEYS]])
            x_scaled = self.scaler.transform(x_vec)
            p_final = float(self.model.predict_proba(x_scaled)[0, 1])
            mode = "meta_learner"
        else:
            p_final = self._weighted_average(scores)
            mode = "weighted_average"

        label = 1 if p_final >= self.threshold else 0

        # Confidence tier
        if p_final >= 0.80:
            confidence = "High"
        elif p_final >= 0.60:
            confidence = "Moderate"
        elif p_final >= 0.40:
            confidence = "Low"
        else:
            confidence = "Very Low"

        return {
            "P_final":     p_final,
            "probability": p_final,   # Alias for backward compatibility
            "label":       label,
            "confidence":  confidence,
            "fusion_mode": mode
        }

    def fuse_all_sites(self, site_list: list) -> list:
        """
        Fuse scores for a full list of candidate sites.

        Args:
            site_list: List of dicts; each must contain P_hybrid, P_motif,
                       P_evidence, P_conserv, P_struct keys.

        Returns:
            Enriched list with P_final, label, confidence, fusion_mode added.
        """
        results = []
        for site in site_list:
            fusion = self.fuse(site)
            enriched = dict(site)
            enriched.update(fusion)
            results.append(enriched)

        positives = sum(1 for r in results if r["label"] == 1)
        print(f"[Decision Agent] Fused {len(results)} sites. "
              f"Predicted positive: {positives} | "
              f"Mode: {results[0]['fusion_mode'] if results else 'N/A'}")
        return results

    # ── Evaluation Metrics ────────────────────────────────────────────────────

    def evaluate(self, site_list: list) -> dict:
        """
        Compute all publication metrics comparing P_hybrid alone vs P_final.

        Args:
            site_list: Sites with true_label, P_hybrid, and P_final already set.

        Returns:
            dict with metrics for both HybridARGME and Agentic HybridARGME.
        """
        y_true = [s["true_label"] for s in site_list if "true_label" in s]
        if not y_true:
            return {"error": "No true_label found in site_list for evaluation."}

        # HybridARGME alone
        y_hybrid = [(1 if s.get("P_hybrid", 0) >= 0.5 else 0) for s in site_list if "true_label" in s]
        p_hybrid_proba = [s.get("P_hybrid", 0.5) for s in site_list if "true_label" in s]

        # Agentic HybridARGME
        y_agentic = [s.get("label", 0) for s in site_list if "true_label" in s]
        p_final_proba = [s.get("P_final", 0.5) for s in site_list if "true_label" in s]

        def _metrics(y_true, y_pred, y_proba):
            return {
                "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
                "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
                "AUC":       round(roc_auc_score(y_true, y_proba), 4),
                "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
                "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            }

        return {
            "HybridARGME_standalone": _metrics(y_true, y_hybrid, p_hybrid_proba),
            "Agentic_HybridARGME":    _metrics(y_true, y_agentic, p_final_proba),
        }

    # ── Orchestrator Interface ─────────────────────────────────────────────────

    def execute(self, data: dict) -> dict:
        """
        Orchestrator-compatible interface.

        Args:
            data: {"sites": list_of_enriched_site_dicts}
        Returns:
            {"final_predictions": list}
        """
        sites = data.get("sites", [])
        if not sites:
            return {"error": "No sites provided to DecisionAgent.", "final_predictions": []}

        final = self.fuse_all_sites(sites)
        return {"final_predictions": final}


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = DecisionAgent()

    test_sites = [
        {"position": 34,  "residue": "R", "motif": "XXXRGGRVLSX",
         "P_hybrid": 0.85, "P_motif": 0.90, "P_struct": 0.50,
         "P_conserv": 1.00, "P_evidence": 0.78},
        {"position": 89,  "residue": "K", "motif": "TGKGRGTAAXX",
         "P_hybrid": 0.55, "P_motif": 0.40, "P_struct": 0.50,
         "P_conserv": 0.50, "P_evidence": 0.22},
        {"position": 120, "residue": "R", "motif": "MAALEDRVVTX",
         "P_hybrid": 0.30, "P_motif": 0.10, "P_struct": 0.50,
         "P_conserv": 0.25, "P_evidence": 0.10},
    ]

    result = agent.execute({"sites": test_sites})
    for s in result["final_predictions"]:
        label_str = "✅ METHYLATED" if s["label"] == 1 else "❌ NOT METHYLATED"
        print(f"  Pos {s['position']} {s['residue']}: "
              f"P_final={s['P_final']:.3f}  [{s['confidence']}]  {label_str}")

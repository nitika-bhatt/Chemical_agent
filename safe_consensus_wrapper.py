"""Auditable safeguard around the trained consensus agent.

The safeguard prevents a strong HybridARGME prediction from being overturned
unless every secondary signal is consistently negative.
"""


class SafeConsensusWrapper:
    def __init__(self, base_consensus_agent,
                 strong_plm_threshold: float = 0.60,
                 negative_evidence_ceiling: float = 0.30,
                 log_overrides: bool = True):
        self.base = base_consensus_agent
        self.strong_plm_threshold = strong_plm_threshold
        self.negative_evidence_ceiling = negative_evidence_ceiling
        self.log_overrides = log_overrides

    @property
    def threshold(self):
        return self.base.threshold

    @property
    def weights(self):
        return self.base.weights

    def evaluate(self, site_list: list) -> dict:
        return self.base.evaluate(site_list)

    def _all_secondary_evidence_negative(self, site: dict) -> bool:
        secondary_scores = [
            site.get("P_motif", 1.0),
            site.get("P_conserv", 1.0),
            site.get("P_evidence", 1.0),
            site.get("P_struct", 1.0),
        ]
        return all(score < self.negative_evidence_ceiling
                   for score in secondary_scores)

    def _has_independent_support(self, site: dict) -> bool:
        return max(
            site.get("P_evidence", 0.0),
            site.get("P_conserv", 0.0),
        ) >= 0.25

    def execute(self, data: dict) -> dict:
        result = self.base.execute(data)
        final_predictions = result.get("final_predictions", data.get("sites", []))
        overridden = 0

        for site in final_predictions:
            was_negative = site.get("label", 0) == 0
            plm_strong_positive = (
                site.get("P_hybrid", 0.0) >= self.strong_plm_threshold
            )
            if was_negative and plm_strong_positive \
                    and not self._all_secondary_evidence_negative(site) \
                    and self._has_independent_support(site):
                if self.log_overrides:
                    print(
                        "[SafeConsensusWrapper] Overriding fused NEGATIVE -> "
                        f"POSITIVE at position {site.get('position')}: "
                        f"P_hybrid={site.get('P_hybrid'):.4f}, "
                        f"original P_final={site.get('P_final', float('nan')):.4f}."
                    )
                site["label"] = 1
                site["P_final"] = max(
                    site.get("P_final", 0.0), site.get("P_hybrid", 0.0)
                )
                site["probability"] = site["P_final"]
                if site["P_final"] >= 0.80:
                    site["confidence"] = "High"
                elif site["P_final"] >= 0.60:
                    site["confidence"] = "Moderate"
                elif site["P_final"] >= 0.40:
                    site["confidence"] = "Low"
                else:
                    site["confidence"] = "Very Low"
                site["fusion_mode"] = (
                    site.get("fusion_mode", "") + "+safeguard_override"
                )
                overridden += 1

        if overridden and self.log_overrides:
            print(f"[SafeConsensusWrapper] {overridden} site(s) overridden this call.")

        result["final_predictions"] = final_predictions
        return result
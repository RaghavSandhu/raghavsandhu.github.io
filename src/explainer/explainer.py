"""Prediction explainability using SHAP and feature analysis."""

import numpy as np
import shap


class PredictionExplainer:
    """Explains predictions using SHAP values, feature importance,
    and model agreement analysis."""

    def __init__(self, xgb_model, feature_names: list[str]):
        self.xgb_model = xgb_model
        self.feature_names = feature_names
        self._shap_explainer = None
        self._shap_values = None

    def compute_shap_values(self, X: np.ndarray, max_samples: int = 500):
        """Compute SHAP values for XGBoost model predictions.

        Args:
            X: Feature matrix (n_samples, n_features).
            max_samples: Max samples to explain (for speed).
        """
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        self._shap_explainer = shap.TreeExplainer(self.xgb_model.model)
        self._shap_values = self._shap_explainer.shap_values(X_sample)
        return self._shap_values

    def explain_prediction(self, X_single: np.ndarray) -> dict:
        """Explain a single prediction with feature contributions.

        Args:
            X_single: Feature vector of shape (1, n_features) or (n_features,).

        Returns:
            Dict with prediction breakdown: which features pushed toward
            which direction and by how much.
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        explainer = shap.TreeExplainer(self.xgb_model.model)
        shap_vals = explainer.shap_values(X_single)

        # XGBoost multi-class: shap_vals is a list of arrays per class
        pred_class = self.xgb_model.model.predict(X_single)[0]
        pred_proba = self.xgb_model.model.predict_proba(X_single)[0]
        direction_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

        # Get SHAP values for the predicted class
        if isinstance(shap_vals, list):
            class_shap = shap_vals[pred_class][0]
        else:
            class_shap = shap_vals[0]

        # Build feature contribution breakdown
        contributions = []
        for i, (name, val) in enumerate(zip(self.feature_names, class_shap)):
            contributions.append({
                "feature": name,
                "shap_value": float(val),
                "feature_value": float(X_single[0, i]),
                "direction": "positive" if val > 0 else "negative",
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "predicted_direction": direction_map[pred_class],
            "probabilities": {
                "DOWN": float(pred_proba[0]),
                "NEUTRAL": float(pred_proba[1]),
                "UP": float(pred_proba[2]),
            },
            "top_features": contributions[:10],
            "all_contributions": contributions,
        }

    def get_global_feature_importance(self, X: np.ndarray, top_n: int = 20) -> list[dict]:
        """Get global feature importance using mean absolute SHAP values.

        Returns:
            List of dicts with feature name and importance, sorted descending.
        """
        if self._shap_values is None:
            self.compute_shap_values(X)

        shap_vals = self._shap_values
        # For multi-class, average across classes
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals)

        # Mean across samples
        importance = np.mean(mean_abs, axis=0)

        result = []
        for name, imp in sorted(zip(self.feature_names, importance), key=lambda x: -x[1]):
            result.append({"feature": name, "importance": float(imp)})
        return result[:top_n]

    def generate_explanation_text(self, explanation: dict) -> str:
        """Generate a human-readable explanation string."""
        direction = explanation["predicted_direction"]
        probs = explanation["probabilities"]
        top = explanation["top_features"][:5]

        lines = [
            f"Prediction: {direction}",
            f"Confidence — UP: {probs['UP']:.1%}  NEUTRAL: {probs['NEUTRAL']:.1%}  DOWN: {probs['DOWN']:.1%}",
            "",
            "Top contributing factors:",
        ]

        for i, feat in enumerate(top, 1):
            sign = "+" if feat["shap_value"] > 0 else ""
            lines.append(
                f"  {i}. {feat['feature']}: value={feat['feature_value']:.4f}, "
                f"impact={sign}{feat['shap_value']:.4f} ({feat['direction']})"
            )

        return "\n".join(lines)


def compute_model_agreement_report(
    ensemble_predictions: dict,
    ensemble_direction: np.ndarray,
) -> list[dict]:
    """Produce per-sample agreement report from ensemble individual predictions.

    Args:
        ensemble_predictions: Output of EnsemblePredictor.get_individual_predictions().
        ensemble_direction: Output of EnsemblePredictor.predict().

    Returns:
        List of dicts, one per sample (aligned to shortest model output).
    """
    model_names = list(ensemble_predictions.keys())
    min_len = min(len(v["predictions"]) for v in ensemble_predictions.values())
    min_len = min(min_len, len(ensemble_direction))

    report = []
    for i in range(min_len):
        sample = {
            "ensemble_direction": int(ensemble_direction[-min_len + i]),
            "models": {},
            "agreement_count": 0,
        }
        for name in model_names:
            pred = int(ensemble_predictions[name]["predictions"][-min_len + i])
            conf = float(ensemble_predictions[name]["confidence"][-min_len + i])
            sample["models"][name] = {"prediction": pred, "confidence": conf}
            if pred == sample["ensemble_direction"]:
                sample["agreement_count"] += 1

        sample["agreement_ratio"] = sample["agreement_count"] / len(model_names)
        report.append(sample)

    return report

"""Dynamic weighted ensemble of multiple prediction models."""

import numpy as np
from sklearn.metrics import accuracy_score


class EnsemblePredictor:
    """Combines predictions from LSTM, XGBoost, and Transformer.

    Weights are dynamically adjusted based on recent prediction accuracy
    (exponentially weighted rolling accuracy).
    """

    def __init__(self, models: dict, seq_len: int = 20, lookback: int = 50):
        """
        Args:
            models: dict of {"name": model_instance} — each must have
                    predict(), get_confidence(), predict_proba() methods.
            seq_len: Sequence length used by sequential models (LSTM/Transformer).
            lookback: Window size for computing rolling accuracy weights.
        """
        self.models = models
        self.seq_len = seq_len
        self.lookback = lookback
        self.weights: dict[str, float] = {name: 1.0 / len(models) for name in models}
        self._accuracy_history: dict[str, list[float]] = {name: [] for name in models}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted vote across models. Returns direction (-1, 0, 1)."""
        all_probs = self._collect_proba(X)
        if not all_probs:
            return np.zeros(self._output_len(X))

        # Weighted average of probabilities
        combined = np.zeros_like(list(all_probs.values())[0])
        for name, probs in all_probs.items():
            combined += self.weights[name] * probs
        return np.argmax(combined, axis=1) - 1

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble confidence (max of weighted average probabilities)."""
        all_probs = self._collect_proba(X)
        if not all_probs:
            return np.ones(self._output_len(X)) * 0.33

        combined = np.zeros_like(list(all_probs.values())[0])
        for name, probs in all_probs.items():
            combined += self.weights[name] * probs
        return np.max(combined, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted average probabilities [P(down), P(neutral), P(up)]."""
        all_probs = self._collect_proba(X)
        if not all_probs:
            n = self._output_len(X)
            return np.full((n, 3), 1.0 / 3)

        combined = np.zeros_like(list(all_probs.values())[0])
        for name, probs in all_probs.items():
            combined += self.weights[name] * probs
        return combined

    def get_model_agreement(self, X: np.ndarray) -> np.ndarray:
        """Return fraction of models agreeing on the predicted direction (0-1)."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Align lengths to shortest prediction
        min_len = min(len(p) for p in predictions.values())
        aligned = {n: p[-min_len:] for n, p in predictions.items()}

        agreement = np.zeros(min_len)
        ensemble_pred = self.predict(X)[-min_len:]
        for pred in aligned.values():
            agreement += (pred == ensemble_pred).astype(float)
        return agreement / len(self.models)

    def update_weights(self, X: np.ndarray, y_true: np.ndarray):
        """Update model weights based on recent prediction accuracy.

        This is the self-correction mechanism: models that perform poorly
        get down-weighted automatically.
        """
        for name, model in self.models.items():
            preds = model.predict(X)
            min_len = min(len(preds), len(y_true))
            preds = preds[-min_len:]
            actual = y_true[-min_len:]
            acc = accuracy_score(actual, preds)
            self._accuracy_history[name].append(acc)

        # Exponentially weighted recent accuracy
        for name in self.models:
            history = self._accuracy_history[name][-self.lookback:]
            if history:
                decay = np.exp(np.linspace(-2, 0, len(history)))
                self.weights[name] = float(np.average(history, weights=decay))

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_individual_predictions(self, X: np.ndarray) -> dict:
        """Return each model's prediction, confidence, and weight."""
        result = {}
        for name, model in self.models.items():
            preds = model.predict(X)
            conf = model.get_confidence(X)
            result[name] = {
                "predictions": preds,
                "confidence": conf,
                "weight": self.weights[name],
            }
        return result

    def _collect_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        all_probs = {}
        for name, model in self.models.items():
            probs = model.predict_proba(X)
            all_probs[name] = probs

        if not all_probs:
            return {}

        # Align to shortest output (sequential models produce fewer samples)
        min_len = min(p.shape[0] for p in all_probs.values())
        return {name: probs[-min_len:] for name, probs in all_probs.items()}

    def _output_len(self, X: np.ndarray) -> int:
        return max(0, len(X) - self.seq_len)

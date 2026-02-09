"""XGBoost model for tabular feature-based prediction."""

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


class XGBoostModel:
    """XGBoost wrapper for direction classification (3-class: down/neutral/up)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            random_state=42,
        )
        self.train_metrics: dict = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        # Map labels -1,0,1 -> 0,1,2 for XGBoost
        y_tr = y_train.astype(int) + 1
        y_v = y_val.astype(int) + 1

        self.model.fit(
            X_train, y_tr,
            eval_set=[(X_val, y_v)],
            verbose=False,
        )

        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        train_acc = accuracy_score(y_tr, train_preds)
        val_acc = accuracy_score(y_v, val_preds)

        report = classification_report(
            y_v, val_preds,
            target_names=["Down", "Neutral", "Up"],
            output_dict=True,
        )

        self.train_metrics = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_iteration": self.model.best_iteration,
            "classification_report": report,
        }
        return self.train_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted direction (-1, 0, 1)."""
        preds = self.model.predict(X)
        return preds.astype(int) - 1

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return confidence (max class probability) per sample."""
        probs = self.model.predict_proba(X)
        return np.max(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities [P(down), P(neutral), P(up)]."""
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature importance dict sorted descending."""
        importance = self.model.feature_importances_
        pairs = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        return dict(pairs)

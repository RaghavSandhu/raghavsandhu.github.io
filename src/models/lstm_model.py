"""LSTM model for sequential price prediction.

Uses PyTorch if available, falls back to sklearn MLPClassifier otherwise.
The sklearn fallback flattens sequences into feature vectors to approximate
sequential learning via a deep MLP with multiple hidden layers.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from src.data.features import create_sequences

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── PyTorch LSTM (used when torch is available) ──────────────────────

if HAS_TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, num_classes),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


class LSTMModel:
    """LSTM wrapper for direction classification (3-class: down/neutral/up)."""

    def __init__(self, hidden_size=128, num_layers=2, seq_len=20,
                 lr=1e-3, epochs=50, batch_size=64):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.train_metrics: dict = {}
        self._use_torch = HAS_TORCH

    def train(self, X_train, y_train, X_val, y_val):
        if self._use_torch:
            return self._train_torch(X_train, y_train, X_val, y_val)
        return self._train_sklearn(X_train, y_train, X_val, y_val)

    def predict(self, X):
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.seq_len)
        if self._use_torch:
            probs = self._predict_proba_torch(X_seq)
        else:
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            probs = self.model.predict_proba(X_flat)
        return np.argmax(probs, axis=1) - 1

    def get_confidence(self, X):
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.seq_len)
        if self._use_torch:
            probs = self._predict_proba_torch(X_seq)
        else:
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            probs = self.model.predict_proba(X_flat)
        return np.max(probs, axis=1)

    def predict_proba(self, X):
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.seq_len)
        if self._use_torch:
            return self._predict_proba_torch(X_seq)
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        return self.model.predict_proba(X_flat)

    # ── sklearn fallback ──────────────────────────────────────────────

    def _train_sklearn(self, X_train, y_train, X_val, y_val):
        X_seq, y_seq = create_sequences(X_train, y_train, self.seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, self.seq_len)

        y_seq = y_seq.astype(int) + 1
        y_val_seq = y_val_seq.astype(int) + 1

        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        X_val_flat = X_val_seq.reshape(X_val_seq.shape[0], -1)

        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=self.epochs,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=self.lr,
            batch_size=min(self.batch_size, len(X_flat)),
            random_state=42,
        )
        self.model.fit(X_flat, y_seq)

        train_acc = accuracy_score(y_seq, self.model.predict(X_flat))
        val_acc = accuracy_score(y_val_seq, self.model.predict(X_val_flat))
        loss_curve = self.model.loss_curve_ if hasattr(self.model, "loss_curve_") else []

        self.train_metrics = {
            "best_val_acc": val_acc,
            "final_train_loss": loss_curve[-1] if loss_curve else 0.0,
            "final_val_loss": 1.0 - val_acc,
            "history": {
                "train_loss": list(loss_curve),
                "val_loss": [1.0 - val_acc] * len(loss_curve),
                "val_acc": [val_acc] * len(loss_curve),
            },
            "backend": "sklearn",
        }
        return self.train_metrics

    # ── PyTorch backend ───────────────────────────────────────────────

    def _train_torch(self, X_train, y_train, X_val, y_val):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_seq, y_seq = create_sequences(X_train, y_train, self.seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, self.seq_len)

        y_seq = y_seq.astype(int) + 1
        y_val_seq = y_val_seq.astype(int) + 1

        input_size = X_seq.shape[2]
        self.model = _LSTMNet(input_size, self.hidden_size, self.num_layers, 3).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.FloatTensor(X_seq), torch.LongTensor(y_seq))
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val_acc = 0.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)

            val_loss, val_acc = self._eval_torch(X_val_seq, y_val_seq, criterion, device)
            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()

        self.train_metrics = {
            "best_val_acc": best_val_acc,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "history": history,
            "backend": "torch",
        }
        return self.train_metrics

    def _predict_proba_torch(self, X_seq):
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X_seq).to(device)
            logits = self.model(t)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def _eval_torch(self, X_seq, y_seq, criterion, device):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X_seq).to(device)
            yt = torch.LongTensor(y_seq).to(device)
            logits = self.model(t)
            loss = criterion(logits, yt).item()
            acc = (logits.argmax(dim=1) == yt).float().mean().item()
        return loss, acc

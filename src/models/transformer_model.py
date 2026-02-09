"""Temporal Transformer model for time-series prediction.

Uses PyTorch if available, falls back to sklearn GradientBoostingClassifier
with sequence-aware feature engineering otherwise.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.data.features import create_sequences

try:
    import math
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=200):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class _TemporalTransformer(nn.Module):
        def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc = _PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            return self.classifier(x[:, -1, :])


class TransformerModel:
    """Temporal Transformer wrapper for direction classification."""

    def __init__(self, d_model=64, nhead=4, num_layers=2, seq_len=20,
                 lr=1e-3, epochs=50, batch_size=64):
        self.d_model = d_model
        self.nhead = nhead
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
            X_aug = self._augment_sequences(X_seq)
            probs = self.model.predict_proba(X_aug)
        return np.argmax(probs, axis=1) - 1

    def get_confidence(self, X):
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.seq_len)
        if self._use_torch:
            probs = self._predict_proba_torch(X_seq)
        else:
            X_aug = self._augment_sequences(X_seq)
            probs = self.model.predict_proba(X_aug)
        return np.max(probs, axis=1)

    def predict_proba(self, X):
        X_seq, _ = create_sequences(X, np.zeros(len(X)), self.seq_len)
        if self._use_torch:
            return self._predict_proba_torch(X_seq)
        X_aug = self._augment_sequences(X_seq)
        return self.model.predict_proba(X_aug)

    # ── Sequence-aware feature augmentation for sklearn ───────────────

    def _augment_sequences(self, X_seq):
        """Create attention-like features from sequences.

        Instead of just flattening, we compute temporal statistics that
        capture what a transformer's attention mechanism would learn:
        - Latest values (most recent timestep)
        - Mean/std across the sequence
        - Trend (difference between first/last half)
        - Weighted recency (exponentially weighted features)
        """
        n_samples, seq_len, n_features = X_seq.shape
        latest = X_seq[:, -1, :]
        mean_feat = X_seq.mean(axis=1)
        std_feat = X_seq.std(axis=1)

        half = seq_len // 2
        first_half = X_seq[:, :half, :].mean(axis=1)
        second_half = X_seq[:, half:, :].mean(axis=1)
        trend = second_half - first_half

        weights = np.exp(np.linspace(-2, 0, seq_len))
        weights /= weights.sum()
        weighted = np.einsum("ijk,j->ik", X_seq, weights)

        return np.hstack([latest, mean_feat, std_feat, trend, weighted])

    def _train_sklearn(self, X_train, y_train, X_val, y_val):
        X_seq, y_seq = create_sequences(X_train, y_train, self.seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, self.seq_len)

        y_seq = y_seq.astype(int) + 1
        y_val_seq = y_val_seq.astype(int) + 1

        X_aug = self._augment_sequences(X_seq)
        X_val_aug = self._augment_sequences(X_val_seq)

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(X_aug, y_seq)

        train_acc = accuracy_score(y_seq, self.model.predict(X_aug))
        val_acc = accuracy_score(y_val_seq, self.model.predict(X_val_aug))

        self.train_metrics = {
            "best_val_acc": val_acc,
            "final_train_loss": 1.0 - train_acc,
            "final_val_loss": 1.0 - val_acc,
            "history": {
                "train_loss": [1.0 - train_acc],
                "val_loss": [1.0 - val_acc],
                "val_acc": [val_acc],
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
        self.model = _TemporalTransformer(
            input_size, self.d_model, self.nhead, self.num_layers, 3,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)
            scheduler.step()

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

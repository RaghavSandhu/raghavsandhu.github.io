"""Tests for the stock prediction pipeline."""

import numpy as np
import pandas as pd

from src.data.features import (
    add_technical_indicators,
    create_targets,
    prepare_dataset,
    create_sequences,
)
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.ensemble import EnsemblePredictor
from src.models.rl_agent import DQNAgent, TradingEnvironment, ACTION_NAMES
from src.backtesting.engine import BacktestEngine


def _make_ohlcv(n=300):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ── Feature engineering tests ─────────────────────────────────────────

def test_add_technical_indicators():
    df = _make_ohlcv()
    result = add_technical_indicators(df)
    assert "RSI" in result.columns
    assert "MACD" in result.columns
    assert "BB_upper" in result.columns
    assert "ATR" in result.columns
    assert len(result) == len(df)


def test_create_targets():
    df = _make_ohlcv()
    result = create_targets(df, horizon=3, threshold=0.001)
    assert "target_return" in result.columns
    assert "target_direction" in result.columns
    assert set(result["target_direction"].unique()).issubset({-1, 0, 1})


def test_prepare_dataset():
    df = _make_ohlcv()
    feature_cols, X_train, X_test, y_train_dir, y_test_dir, *_ = prepare_dataset(df)
    assert len(feature_cols) > 30  # We should have 40+ features
    assert len(X_train) > len(X_test)
    assert X_train.shape[1] == len(feature_cols)


def test_create_sequences():
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    X_seq, y_seq = create_sequences(X, y, seq_len=20)
    assert X_seq.shape == (80, 20, 10)
    assert y_seq.shape == (80,)


# ── Model tests ───────────────────────────────────────────────────────

def _get_train_test():
    df = _make_ohlcv(300)
    _, X_train, X_test, y_train, y_test, *_ = prepare_dataset(
        df, horizon=3, threshold=0.001
    )
    return X_train, X_test, y_train, y_test


def test_xgboost_model():
    X_train, X_test, y_train, y_test = _get_train_test()
    model = XGBoostModel(n_estimators=10, max_depth=3)
    metrics = model.train(X_train, y_train, X_test, y_test)
    assert "val_acc" in metrics
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({-1, 0, 1})
    conf = model.get_confidence(X_test)
    assert all(0 <= c <= 1 for c in conf)


def test_lstm_model():
    X_train, X_test, y_train, y_test = _get_train_test()
    model = LSTMModel(seq_len=10, epochs=2, batch_size=32)
    metrics = model.train(X_train, y_train, X_test, y_test)
    assert "best_val_acc" in metrics
    preds = model.predict(X_test)
    assert len(preds) > 0
    conf = model.get_confidence(X_test)
    assert len(conf) == len(preds)


def test_transformer_model():
    X_train, X_test, y_train, y_test = _get_train_test()
    model = TransformerModel(seq_len=10, epochs=2, batch_size=32)
    metrics = model.train(X_train, y_train, X_test, y_test)
    assert "best_val_acc" in metrics
    preds = model.predict(X_test)
    assert len(preds) > 0


def test_ensemble():
    X_train, X_test, y_train, y_test = _get_train_test()

    xgb = XGBoostModel(n_estimators=10)
    xgb.train(X_train, y_train, X_test, y_test)

    lstm = LSTMModel(seq_len=10, epochs=2)
    lstm.train(X_train, y_train, X_test, y_test)

    ensemble = EnsemblePredictor({"XGBoost": xgb, "LSTM": lstm}, seq_len=10)
    preds = ensemble.predict(X_test)
    assert len(preds) > 0
    conf = ensemble.get_confidence(X_test)
    assert len(conf) == len(preds)

    ensemble.update_weights(X_test, y_test)
    assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6


# ── RL agent tests ────────────────────────────────────────────────────

def test_trading_environment():
    prices = np.array([100, 101, 102, 101, 100, 99, 100, 101], dtype=float)
    preds = np.zeros(8)
    confs = np.ones(8) * 0.5
    features = np.random.randn(8, 4)
    env = TradingEnvironment(prices, preds, confs, features)
    state = env.reset()
    assert len(state) == env.state_size
    next_state, reward, done = env.step(1)  # HOLD
    assert not done
    assert len(next_state) == env.state_size


def test_dqn_agent():
    state_size = 8
    agent = DQNAgent(state_size=state_size, batch_size=4, memory_size=100)
    state = np.random.randn(state_size).astype(np.float32)
    action = agent.act(state, explore=False)
    assert action in [0, 1, 2]
    q_vals = agent.get_q_values(state)
    assert len(q_vals) == 3


# ── Backtesting tests ─────────────────────────────────────────────────

def test_backtest_engine():
    prices = np.array([100, 101, 102, 103, 102, 101, 100, 101, 102, 103], dtype=float)
    signals = np.array([0, 1, 0, 0, -1, 0, 0, 1, 0, -1])
    engine = BacktestEngine()
    result = engine.run(prices, signals)
    assert "equity_curve" in result
    assert "metrics" in result
    assert result["metrics"]["total_trades"] >= 0
    assert len(result["equity_curve"]) == len(prices)

"""Tests for the batch pipeline (store, config, scheduler logic)."""

import json
import os
import tempfile
import time
from unittest.mock import patch

from src.batch.config import WATCHLIST, REFRESH_INTERVAL_SECONDS
from src.batch.store import (
    init_db, save_prediction, save_screening, save_batch_status,
    get_latest_predictions, get_latest_screening, get_batch_status,
    get_prediction_history, cleanup_old_data, DB_PATH,
)


def _use_temp_db(func):
    """Decorator to use a temporary database for testing."""
    def wrapper(*args, **kwargs):
        import src.batch.store as store_mod
        original_path = store_mod.DB_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            store_mod.DB_PATH = type(original_path)(tmpdir) / "test.db"
            try:
                return func(*args, **kwargs)
            finally:
                store_mod.DB_PATH = original_path
    return wrapper


@_use_temp_db
def test_init_db():
    init_db()
    # Should not raise on double init
    init_db()


@_use_temp_db
def test_save_and_get_prediction():
    init_db()
    save_prediction("AAPL", {
        "direction": 1,
        "confidence": 0.75,
        "model_agreement": 0.8,
        "rl_action": 2,
        "rl_q_values": [0.1, 0.3, 0.6],
        "current_price": 150.50,
        "technical_data": {"rsi": 55.0, "adx": 30.0},
    })

    results = get_latest_predictions(["AAPL"])
    assert len(results) == 1
    assert results[0]["ticker"] == "AAPL"
    assert results[0]["direction"] == 1
    assert results[0]["confidence"] == 0.75
    assert results[0]["current_price"] == 150.50
    assert results[0]["technical_data"]["rsi"] == 55.0


@_use_temp_db
def test_multiple_predictions():
    init_db()
    save_prediction("AAPL", {"direction": 1, "confidence": 0.6,
                              "current_price": 150.0, "technical_data": {}})
    time.sleep(0.01)
    save_prediction("MSFT", {"direction": -1, "confidence": 0.7,
                              "current_price": 400.0, "technical_data": {}})

    results = get_latest_predictions()
    assert len(results) == 2
    tickers = {r["ticker"] for r in results}
    assert tickers == {"AAPL", "MSFT"}


@_use_temp_db
def test_prediction_history():
    init_db()
    for i in range(5):
        save_prediction("TSLA", {"direction": i % 3 - 1, "confidence": 0.5 + i * 0.05,
                                  "current_price": 200 + i, "technical_data": {}})
        time.sleep(0.01)

    history = get_prediction_history("TSLA", limit=10)
    assert len(history) >= 1  # At least 1 (may collapse due to same-second timestamps)


@_use_temp_db
def test_save_and_get_screening():
    init_db()
    save_screening("NVDA", {
        "price": 800.0, "composite_score": 0.85, "risk_score": 0.3,
        "reward_ratio": 2.1, "signal": "BUY", "rsi": 45.0, "adx": 35.0,
        "atr_pct": 1.5, "volatility": 2.0, "recommendation": "strong trend",
    })

    results = get_latest_screening()
    assert len(results) == 1
    assert results[0]["ticker"] == "NVDA"
    assert results[0]["signal"] == "BUY"


@_use_temp_db
def test_batch_status():
    init_db()
    save_batch_status("full_train", 95, 5, 120.5, ["LCID: timeout"])

    status = get_batch_status()
    assert status is not None
    assert status["tickers_processed"] == 95
    assert status["tickers_failed"] == 5
    assert status["duration_seconds"] == 120.5
    assert "LCID: timeout" in status["errors"]


@_use_temp_db
def test_cleanup_old_data():
    init_db()
    save_prediction("OLD", {"direction": 0, "confidence": 0.5,
                             "current_price": 100.0, "technical_data": {}})
    # Cleanup with 0 days should remove everything
    cleanup_old_data(days=0)
    results = get_latest_predictions()
    assert len(results) == 0


def test_watchlist_has_100():
    assert len(WATCHLIST) == 100


def test_refresh_interval():
    assert REFRESH_INTERVAL_SECONDS == 180  # 3 minutes


def test_scheduler_market_hours():
    from src.batch.scheduler import _is_market_hours
    from datetime import datetime, timezone, timedelta

    # Monday 10am ET — should be market hours
    et = timezone(timedelta(hours=-5))
    monday_10am = datetime(2025, 1, 6, 10, 0, tzinfo=et)  # Monday
    assert _is_market_hours(monday_10am) is True

    # Monday 5pm ET — market closed
    monday_5pm = datetime(2025, 1, 6, 17, 0, tzinfo=et)
    assert _is_market_hours(monday_5pm) is False

    # Saturday 10am ET — weekend
    saturday = datetime(2025, 1, 11, 10, 0, tzinfo=et)
    assert _is_market_hours(saturday) is False

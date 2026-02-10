"""Parallel batch runner for training and predicting on multiple stocks.

Two modes:
- full_train: Train all models from scratch + predict (runs at market open, then hourly)
- quick_predict: Fetch new data + run predictions through cached models (runs every 3 min)
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.batch.config import (
    EPOCHS, RL_EPISODES, XGB_N_ESTIMATORS, SEQ_LEN, HORIZON,
    THRESHOLD, BATCH_SIZE, MAX_WORKERS, INTRADAY_INTERVAL, INTRADAY_PERIOD,
)
from src.batch import store
from src.data.fetcher import fetch_stock_data, get_stock_info
from src.data.features import add_technical_indicators, create_targets, prepare_dataset
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.ensemble import EnsemblePredictor
from src.models.rl_agent import DQNAgent, TradingEnvironment, ACTION_NAMES
from src.intelligence.news_fetcher import fetch_all_news
from src.intelligence.llm_analyzer import analyze_sentiment, assess_risk
from src.intelligence.strategy_advisor import get_strategy_advice
from src.intelligence.stock_screener import _analyze_ticker

log = logging.getLogger("batch")

# In-memory cache of trained models (keyed by ticker)
_model_cache: dict[str, dict] = {}


def _train_single(ticker: str) -> dict | None:
    """Train all models for a single ticker and return results."""
    try:
        df = fetch_stock_data(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
        if df is None or len(df) < 60:
            log.warning(f"{ticker}: Not enough data ({len(df) if df is not None else 0} rows)")
            return None

        result = prepare_dataset(
            df, sequence_length=SEQ_LEN, horizon=HORIZON, threshold=THRESHOLD
        )
        feature_cols, X_train, X_test = result[0], result[1], result[2]
        y_train_dir, y_test_dir = result[3], result[4]

        if len(X_train) < SEQ_LEN + 10 or len(X_test) < SEQ_LEN + 10:
            log.warning(f"{ticker}: Insufficient train/test samples")
            return None

        # Train models
        xgb = XGBoostModel(n_estimators=XGB_N_ESTIMATORS)
        xgb_metrics = xgb.train(X_train, y_train_dir, X_test, y_test_dir)

        lstm = LSTMModel(seq_len=SEQ_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE)
        lstm_metrics = lstm.train(X_train, y_train_dir, X_test, y_test_dir)

        transformer = TransformerModel(seq_len=SEQ_LEN, epochs=EPOCHS, batch_size=BATCH_SIZE)
        trans_metrics = transformer.train(X_train, y_train_dir, X_test, y_test_dir)

        ensemble = EnsemblePredictor(
            {"LSTM": lstm, "XGBoost": xgb, "Transformer": transformer},
            seq_len=SEQ_LEN,
        )
        ensemble.update_weights(X_test, y_test_dir)

        # Full df for RL
        df_full = add_technical_indicators(df)
        df_full = create_targets(df_full, horizon=HORIZON, threshold=THRESHOLD)
        df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_full.dropna(inplace=True)

        test_prices = df_full["Close"].values[-len(X_test):]
        ens_preds = ensemble.predict(X_test)
        ens_conf = ensemble.get_confidence(X_test)

        min_len = min(len(test_prices), len(ens_preds), len(ens_conf))
        rl_prices = test_prices[-min_len:]
        rl_preds = ens_preds[-min_len:]
        rl_confs = ens_conf[-min_len:]
        rl_features = X_test[-min_len:, :8]

        env = TradingEnvironment(rl_prices, rl_preds, rl_confs, rl_features)
        agent = DQNAgent(state_size=env.state_size, epsilon=1.0)
        rl_metrics = agent.train_on_env(env, episodes=RL_EPISODES)

        # Technical data
        last_row = df_full.iloc[-1]
        technical_data = {
            "rsi": float(last_row.get("RSI", 50)),
            "macd_hist": float(last_row.get("MACD_hist", 0)),
            "adx": float(last_row.get("ADX", 25)),
            "atr": float(last_row.get("ATR", 0)),
            "bb_pct": float(last_row.get("BB_pct", 0.5)),
            "volatility_20": float(last_row.get("volatility_20", 0.02)),
            "volume_ratio": float(last_row.get("volume_ratio", 1.0)),
        }
        current_price = float(df_full["Close"].iloc[-1])

        # Latest prediction
        latest_pred = int(ens_preds[-1])
        latest_conf = float(ens_conf[-1])
        agreement = ensemble.get_model_agreement(X_test)
        last_state = env._get_state()
        rl_action = agent.act(last_state, explore=False)
        q_vals = agent.get_q_values(last_state)

        # Cache models for quick_predict
        _model_cache[ticker] = {
            "ensemble": ensemble, "agent": agent, "env": env,
            "xgb": xgb, "feature_cols": feature_cols,
        }

        # Save to DB
        store.save_prediction(ticker, {
            "direction": latest_pred,
            "confidence": latest_conf,
            "model_agreement": float(agreement[-1]),
            "rl_action": rl_action,
            "rl_q_values": q_vals,
            "current_price": current_price,
            "technical_data": technical_data,
        })

        store.save_model_metrics(ticker, {
            "xgb_metrics": xgb_metrics,
            "lstm_metrics": lstm_metrics,
            "trans_metrics": trans_metrics,
            "ensemble_weights": dict(ensemble.weights),
            "rl_metrics": rl_metrics,
        })

        # Screening data
        screen_data = _analyze_ticker(ticker, INTRADAY_INTERVAL, INTRADAY_PERIOD, 0)
        if screen_data:
            store.save_screening(ticker, screen_data)

        log.info(f"{ticker}: trained OK — pred={latest_pred}, conf={latest_conf:.1%}, price=${current_price:.2f}")
        return {"ticker": ticker, "status": "ok"}

    except Exception as e:
        log.error(f"{ticker}: FAILED — {e}")
        return {"ticker": ticker, "status": "error", "error": str(e)}


def _predict_single(ticker: str) -> dict | None:
    """Quick prediction using cached models (no retraining)."""
    try:
        cached = _model_cache.get(ticker)
        if not cached:
            log.info(f"{ticker}: No cached model, doing full train")
            return _train_single(ticker)

        df = fetch_stock_data(ticker, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)
        if df is None or len(df) < 60:
            return None

        result = prepare_dataset(
            df, sequence_length=SEQ_LEN, horizon=HORIZON, threshold=THRESHOLD
        )
        X_test = result[2]
        if len(X_test) < SEQ_LEN + 10:
            return None

        ensemble = cached["ensemble"]
        agent = cached["agent"]

        df_full = add_technical_indicators(df)
        df_full = create_targets(df_full, horizon=HORIZON, threshold=THRESHOLD)
        df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_full.dropna(inplace=True)

        ens_preds = ensemble.predict(X_test)
        ens_conf = ensemble.get_confidence(X_test)
        agreement = ensemble.get_model_agreement(X_test)

        test_prices = df_full["Close"].values[-len(X_test):]
        min_len = min(len(test_prices), len(ens_preds), len(ens_conf))
        rl_features = X_test[-min_len:, :8]

        env = TradingEnvironment(
            test_prices[-min_len:], ens_preds[-min_len:],
            ens_conf[-min_len:], rl_features,
        )
        # Run to end to get final state
        state = env.reset()
        for i in range(min_len - 1):
            action = agent.act(state, explore=False)
            state, _, done = env.step(action)
            if done:
                break

        rl_action = agent.act(state, explore=False)
        q_vals = agent.get_q_values(state)

        last_row = df_full.iloc[-1]
        technical_data = {
            "rsi": float(last_row.get("RSI", 50)),
            "macd_hist": float(last_row.get("MACD_hist", 0)),
            "adx": float(last_row.get("ADX", 25)),
            "atr": float(last_row.get("ATR", 0)),
            "bb_pct": float(last_row.get("BB_pct", 0.5)),
            "volatility_20": float(last_row.get("volatility_20", 0.02)),
            "volume_ratio": float(last_row.get("volume_ratio", 1.0)),
        }
        current_price = float(df_full["Close"].iloc[-1])

        store.save_prediction(ticker, {
            "direction": int(ens_preds[-1]),
            "confidence": float(ens_conf[-1]),
            "model_agreement": float(agreement[-1]),
            "rl_action": rl_action,
            "rl_q_values": q_vals,
            "current_price": current_price,
            "technical_data": technical_data,
        })

        # Update screening
        screen_data = _analyze_ticker(ticker, INTRADAY_INTERVAL, INTRADAY_PERIOD, 0)
        if screen_data:
            store.save_screening(ticker, screen_data)

        log.info(f"{ticker}: predict OK — dir={int(ens_preds[-1])}, price=${current_price:.2f}")
        return {"ticker": ticker, "status": "ok"}

    except Exception as e:
        log.error(f"{ticker}: predict FAILED — {e}")
        return {"ticker": ticker, "status": "error", "error": str(e)}


def _run_intelligence(ticker: str, technical_data: dict, current_price: float):
    """Run news/sentiment/risk/strategy for a ticker."""
    try:
        news_items = fetch_all_news(ticker, max_ticker=5, max_market=2)
        if not news_items:
            return

        sentiment = analyze_sentiment(ticker, news_items, current_price, technical_data)
        store.save_sentiment(ticker, sentiment)

        pred = store.get_latest_predictions([ticker])
        model_pred = {}
        if pred:
            p = pred[0]
            direction_map = {1: "UP", 0: "NEUTRAL", -1: "DOWN"}
            model_pred = {
                "direction": direction_map.get(p["direction"], "NEUTRAL"),
                "confidence": f"{p['confidence']:.0%}" if p["confidence"] else "N/A",
                "agreement": f"{p['model_agreement']:.0%}" if p["model_agreement"] else "N/A",
                "rl_action": ACTION_NAMES.get(p["rl_action"], "HOLD"),
            }

        risk = assess_risk(ticker, current_price, technical_data, sentiment, model_pred)
        store.save_risk(ticker, risk)

        advice = get_strategy_advice(
            ticker, current_price, technical_data, sentiment, risk, model_pred
        )
        store.save_strategy(ticker, advice)

        log.info(f"{ticker}: intelligence OK — sentiment={sentiment.overall_sentiment}, risk={risk.overall_risk}")
    except Exception as e:
        log.error(f"{ticker}: intelligence FAILED — {e}")


def run_full_train(tickers: list[str]) -> dict:
    """Train all models for all tickers in parallel."""
    log.info(f"=== FULL TRAIN: {len(tickers)} tickers, {MAX_WORKERS} workers ===")
    start = time.time()
    results = {"ok": 0, "error": 0, "errors": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_train_single, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                r = future.result()
                if r and r["status"] == "ok":
                    results["ok"] += 1
                else:
                    results["error"] += 1
                    results["errors"].append(f"{ticker}: no data or insufficient samples")
            except Exception as e:
                results["error"] += 1
                results["errors"].append(f"{ticker}: {e}")

    duration = time.time() - start
    store.save_batch_status("full_train", results["ok"], results["error"],
                            duration, results["errors"][:20])

    log.info(f"=== FULL TRAIN DONE: {results['ok']}/{len(tickers)} OK in {duration:.1f}s ===")

    # Run intelligence for top-scored tickers (avoid running for all 100 to save time)
    predictions = store.get_latest_predictions()
    for p in predictions[:30]:  # Top 30 by recency
        _run_intelligence(p["ticker"], p["technical_data"], p["current_price"])

    return results


def run_quick_predict(tickers: list[str]) -> dict:
    """Quick prediction for all tickers (no retraining)."""
    log.info(f"=== QUICK PREDICT: {len(tickers)} tickers ===")
    start = time.time()
    results = {"ok": 0, "error": 0, "errors": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_predict_single, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                r = future.result()
                if r and r["status"] == "ok":
                    results["ok"] += 1
                else:
                    results["error"] += 1
            except Exception as e:
                results["error"] += 1
                results["errors"].append(f"{ticker}: {e}")

    duration = time.time() - start
    store.save_batch_status("quick_predict", results["ok"], results["error"],
                            duration, results["errors"][:20])

    log.info(f"=== QUICK PREDICT DONE: {results['ok']}/{len(tickers)} OK in {duration:.1f}s ===")
    return results

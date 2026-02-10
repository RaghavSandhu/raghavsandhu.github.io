"""SQLite store for batch prediction results.

Stores predictions, model metrics, risk/sentiment, and screening results
so the dashboard can read them instantly without recomputing.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.batch.config import DB_PATH, DATA_DIR


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _adapt_numpy(val):
    """Convert numpy types to Python native for SQLite."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


@contextmanager
def get_db():
    """Get a database connection with WAL mode for concurrent reads."""
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                direction INTEGER,
                confidence REAL,
                model_agreement REAL,
                rl_action INTEGER,
                rl_q_values TEXT,
                current_price REAL,
                technical_data TEXT,
                PRIMARY KEY (ticker, timestamp)
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                xgb_metrics TEXT,
                lstm_metrics TEXT,
                trans_metrics TEXT,
                ensemble_weights TEXT,
                rl_metrics TEXT,
                PRIMARY KEY (ticker, timestamp)
            );

            CREATE TABLE IF NOT EXISTS sentiment (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                overall_sentiment TEXT,
                confidence REAL,
                key_drivers TEXT,
                news_impact TEXT,
                risk_level TEXT,
                summary TEXT,
                PRIMARY KEY (ticker, timestamp)
            );

            CREATE TABLE IF NOT EXISTS risk_assessment (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                overall_risk TEXT,
                risk_score REAL,
                volatility_risk TEXT,
                news_risk TEXT,
                technical_risk TEXT,
                position_recommendation TEXT,
                max_loss_estimate TEXT,
                factors TEXT,
                hedging_suggestions TEXT,
                PRIMARY KEY (ticker, timestamp)
            );

            CREATE TABLE IF NOT EXISTS strategy (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                recommended_action TEXT,
                confidence REAL,
                strategies TEXT,
                warnings TEXT,
                summary TEXT,
                PRIMARY KEY (ticker, timestamp)
            );

            CREATE TABLE IF NOT EXISTS screening (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                price REAL,
                composite_score REAL,
                risk_score REAL,
                reward_ratio REAL,
                signal TEXT,
                rsi REAL,
                adx REAL,
                atr_pct REAL,
                volatility REAL,
                recommendation TEXT,
                PRIMARY KEY (ticker, timestamp)
            );

            CREATE TABLE IF NOT EXISTS batch_status (
                run_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tickers_processed INTEGER,
                tickers_failed INTEGER,
                duration_seconds REAL,
                errors TEXT,
                PRIMARY KEY (run_type, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
            CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_screening_ts ON screening(timestamp);
            CREATE INDEX IF NOT EXISTS idx_batch_ts ON batch_status(timestamp);
        """)


def save_prediction(ticker: str, data: dict):
    """Save a prediction result for a ticker."""
    ts = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO predictions
            (ticker, timestamp, direction, confidence, model_agreement,
             rl_action, rl_q_values, current_price, technical_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, ts,
            _adapt_numpy(data.get("direction")),
            _adapt_numpy(data.get("confidence")),
            _adapt_numpy(data.get("model_agreement")),
            _adapt_numpy(data.get("rl_action")),
            json.dumps([_adapt_numpy(v) for v in data.get("rl_q_values", [])]),
            _adapt_numpy(data.get("current_price")),
            json.dumps({k: _adapt_numpy(v) for k, v in data.get("technical_data", {}).items()}),
        ))


def save_model_metrics(ticker: str, data: dict):
    """Save model training metrics."""
    ts = time.time()

    def _clean_metrics(m):
        if m is None:
            return "{}"
        cleaned = {}
        for k, v in m.items():
            if k == "history":
                cleaned[k] = {hk: [_adapt_numpy(x) for x in hv] for hk, hv in v.items()}
            elif k == "classification_report":
                cleaned[k] = {rk: {sk: _adapt_numpy(sv) for sk, sv in rv.items()}
                              if isinstance(rv, dict) else _adapt_numpy(rv)
                              for rk, rv in v.items()}
            else:
                cleaned[k] = _adapt_numpy(v)
        return json.dumps(cleaned)

    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_metrics
            (ticker, timestamp, xgb_metrics, lstm_metrics, trans_metrics,
             ensemble_weights, rl_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, ts,
            _clean_metrics(data.get("xgb_metrics")),
            _clean_metrics(data.get("lstm_metrics")),
            _clean_metrics(data.get("trans_metrics")),
            json.dumps({k: _adapt_numpy(v) for k, v in data.get("ensemble_weights", {}).items()}),
            _clean_metrics(data.get("rl_metrics")),
        ))


def save_sentiment(ticker: str, result):
    """Save sentiment analysis result."""
    ts = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sentiment
            (ticker, timestamp, overall_sentiment, confidence, key_drivers,
             news_impact, risk_level, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, ts,
            result.overall_sentiment,
            _adapt_numpy(result.confidence),
            json.dumps(result.key_drivers),
            result.news_impact,
            result.risk_level,
            result.summary,
        ))


def save_risk(ticker: str, result):
    """Save risk assessment result."""
    ts = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO risk_assessment
            (ticker, timestamp, overall_risk, risk_score, volatility_risk,
             news_risk, technical_risk, position_recommendation,
             max_loss_estimate, factors, hedging_suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, ts,
            result.overall_risk,
            _adapt_numpy(result.risk_score),
            result.volatility_risk,
            result.news_risk,
            result.technical_risk,
            result.position_recommendation,
            result.max_loss_estimate,
            json.dumps(result.factors),
            json.dumps(result.hedging_suggestions),
        ))


def save_strategy(ticker: str, result):
    """Save strategy advice."""
    ts = time.time()
    strategies = []
    for s in result.strategies:
        strategies.append({
            "name": s.name, "description": s.description,
            "entry_conditions": s.entry_conditions,
            "exit_conditions": s.exit_conditions,
            "stop_loss": s.stop_loss, "take_profit": s.take_profit,
            "position_size": s.position_size, "timeframe": s.timeframe,
            "risk_reward": s.risk_reward,
            "confidence": _adapt_numpy(s.confidence),
        })
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO strategy
            (ticker, timestamp, recommended_action, confidence,
             strategies, warnings, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, ts,
            result.recommended_action,
            _adapt_numpy(result.confidence),
            json.dumps(strategies),
            json.dumps(result.warnings),
            result.summary,
        ))


def save_screening(ticker: str, data: dict):
    """Save screening result for a ticker."""
    ts = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO screening
            (ticker, timestamp, price, composite_score, risk_score,
             reward_ratio, signal, rsi, adx, atr_pct, volatility,
             recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, ts,
            _adapt_numpy(data.get("price")),
            _adapt_numpy(data.get("composite_score")),
            _adapt_numpy(data.get("risk_score")),
            _adapt_numpy(data.get("reward_ratio")),
            data.get("signal"),
            _adapt_numpy(data.get("rsi")),
            _adapt_numpy(data.get("adx")),
            _adapt_numpy(data.get("atr_pct")),
            _adapt_numpy(data.get("volatility")),
            data.get("recommendation"),
        ))


def save_batch_status(run_type: str, processed: int, failed: int,
                      duration: float, errors: list):
    """Record a batch run status."""
    ts = time.time()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO batch_status
            (run_type, timestamp, tickers_processed, tickers_failed,
             duration_seconds, errors)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_type, ts, processed, failed, duration, json.dumps(errors)))


# ── Read functions (for dashboard) ───────────────────────────────────

def get_latest_predictions(tickers: list[str] | None = None) -> list[dict]:
    """Get the most recent prediction for each ticker."""
    with get_db() as conn:
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            rows = conn.execute(f"""
                SELECT p.* FROM predictions p
                INNER JOIN (
                    SELECT ticker, MAX(timestamp) as max_ts
                    FROM predictions
                    WHERE ticker IN ({placeholders})
                    GROUP BY ticker
                ) latest ON p.ticker = latest.ticker AND p.timestamp = latest.max_ts
                ORDER BY p.ticker
            """, tickers).fetchall()
        else:
            rows = conn.execute("""
                SELECT p.* FROM predictions p
                INNER JOIN (
                    SELECT ticker, MAX(timestamp) as max_ts
                    FROM predictions GROUP BY ticker
                ) latest ON p.ticker = latest.ticker AND p.timestamp = latest.max_ts
                ORDER BY p.ticker
            """).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            d["rl_q_values"] = json.loads(d["rl_q_values"]) if d["rl_q_values"] else []
            d["technical_data"] = json.loads(d["technical_data"]) if d["technical_data"] else {}
            results.append(d)
        return results


def get_latest_screening() -> list[dict]:
    """Get the most recent screening results."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.* FROM screening s
            INNER JOIN (
                SELECT ticker, MAX(timestamp) as max_ts
                FROM screening GROUP BY ticker
            ) latest ON s.ticker = latest.ticker AND s.timestamp = latest.max_ts
            ORDER BY s.composite_score DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_latest_sentiment(ticker: str) -> dict | None:
    """Get latest sentiment for a ticker."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM sentiment WHERE ticker = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (ticker,)).fetchone()
        if row:
            d = dict(row)
            d["key_drivers"] = json.loads(d["key_drivers"]) if d["key_drivers"] else []
            return d
        return None


def get_latest_risk(ticker: str) -> dict | None:
    """Get latest risk assessment for a ticker."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM risk_assessment WHERE ticker = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (ticker,)).fetchone()
        if row:
            d = dict(row)
            d["factors"] = json.loads(d["factors"]) if d["factors"] else []
            d["hedging_suggestions"] = json.loads(d["hedging_suggestions"]) if d["hedging_suggestions"] else []
            return d
        return None


def get_latest_strategy(ticker: str) -> dict | None:
    """Get latest strategy for a ticker."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM strategy WHERE ticker = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (ticker,)).fetchone()
        if row:
            d = dict(row)
            d["strategies"] = json.loads(d["strategies"]) if d["strategies"] else []
            d["warnings"] = json.loads(d["warnings"]) if d["warnings"] else []
            return d
        return None


def get_latest_model_metrics(ticker: str) -> dict | None:
    """Get latest model metrics for a ticker."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM model_metrics WHERE ticker = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (ticker,)).fetchone()
        if row:
            d = dict(row)
            for key in ["xgb_metrics", "lstm_metrics", "trans_metrics",
                        "ensemble_weights", "rl_metrics"]:
                d[key] = json.loads(d[key]) if d[key] else {}
            return d
        return None


def get_batch_status() -> dict | None:
    """Get the latest batch run status."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM batch_status ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        if row:
            d = dict(row)
            d["errors"] = json.loads(d["errors"]) if d["errors"] else []
            return d
        return None


def get_prediction_history(ticker: str, limit: int = 100) -> list[dict]:
    """Get prediction history for a ticker."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions WHERE ticker = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (ticker, limit)).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["rl_q_values"] = json.loads(d["rl_q_values"]) if d["rl_q_values"] else []
            d["technical_data"] = json.loads(d["technical_data"]) if d["technical_data"] else {}
            results.append(d)
        return list(reversed(results))


def cleanup_old_data(days: int = 7):
    """Remove data older than N days."""
    cutoff = time.time() - days * 86400
    with get_db() as conn:
        for table in ["predictions", "model_metrics", "sentiment",
                      "risk_assessment", "strategy", "screening", "batch_status"]:
            conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))

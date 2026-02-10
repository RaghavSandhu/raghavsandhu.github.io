"""Streamlit dashboard — batch mode.

Reads pre-computed results from SQLite (populated by the scheduler).
Loads instantly — no training on the dashboard side.

Usage:
    streamlit run src/dashboard/batch_app.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.batch.store import (
    init_db, get_latest_predictions, get_latest_screening,
    get_latest_sentiment, get_latest_risk, get_latest_strategy,
    get_latest_model_metrics, get_batch_status, get_prediction_history,
)
from src.batch.config import WATCHLIST
from src.models.rl_agent import ACTION_NAMES

st.set_page_config(page_title="Stock Predictor — 100 Stocks", layout="wide")

DIRECTION_MAP = {1: "UP", 0: "NEUTRAL", -1: "DOWN"}
DIRECTION_COLORS = {1: "#00ff00", 0: "#888888", -1: "#ff4444"}


def main():
    init_db()

    st.title("Stock Trend Predictor — 100 Stocks")

    # Batch status
    status = get_batch_status()
    if status:
        ts = datetime.fromtimestamp(status["timestamp"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Update", ts.strftime("%H:%M:%S"))
        col2.metric("Run Type", status["run_type"].replace("_", " ").title())
        col3.metric("Processed", f"{status['tickers_processed']}/{status['tickers_processed'] + status['tickers_failed']}")
        col4.metric("Duration", f"{status['duration_seconds']:.1f}s")
    else:
        st.warning("No batch data yet. Start the scheduler: `python -m src.batch.scheduler`")
        return

    # Auto-refresh every 60s
    st.caption("Dashboard auto-refreshes every 60 seconds")

    tabs = st.tabs([
        "Overview", "Stock Detail", "Screener",
        "News & Sentiment", "Risk & Strategy", "System",
    ])

    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_detail()
    with tabs[2]:
        render_screener()
    with tabs[3]:
        render_news()
    with tabs[4]:
        render_strategy()
    with tabs[5]:
        render_system()

    # Auto-refresh
    time.sleep(60)
    st.rerun()


def render_overview():
    st.subheader("All Predictions — Latest")

    predictions = get_latest_predictions()
    if not predictions:
        st.info("No predictions yet.")
        return

    rows = []
    for p in predictions:
        rows.append({
            "Ticker": p["ticker"],
            "Price": f"${p['current_price']:.2f}" if p["current_price"] else "N/A",
            "Direction": DIRECTION_MAP.get(p["direction"], "?"),
            "Confidence": f"{p['confidence']:.0%}" if p["confidence"] else "N/A",
            "Agreement": f"{p['model_agreement']:.0%}" if p["model_agreement"] else "N/A",
            "RL Action": ACTION_NAMES.get(p["rl_action"], "?"),
            "RSI": f"{p['technical_data'].get('rsi', 0):.0f}",
            "ADX": f"{p['technical_data'].get('adx', 0):.0f}",
        })

    df = pd.DataFrame(rows)

    def color_direction(val):
        colors = {"UP": "background-color: #1a5c1a", "DOWN": "background-color: #7d2d2d",
                  "NEUTRAL": "background-color: #4a4a2a"}
        return colors.get(val, "")

    def color_action(val):
        colors = {"BUY": "background-color: #1a5c1a", "SELL": "background-color: #7d2d2d",
                  "HOLD": "background-color: #4a4a2a"}
        return colors.get(val, "")

    st.dataframe(
        df.style.map(color_direction, subset=["Direction"])
                .map(color_action, subset=["RL Action"]),
        use_container_width=True, hide_index=True, height=600,
    )

    # Summary stats
    up = sum(1 for p in predictions if p["direction"] == 1)
    down = sum(1 for p in predictions if p["direction"] == -1)
    neutral = sum(1 for p in predictions if p["direction"] == 0)
    buy = sum(1 for p in predictions if p["rl_action"] == 2)
    sell = sum(1 for p in predictions if p["rl_action"] == 0)

    st.markdown("### Market Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("UP", up)
    col2.metric("DOWN", down)
    col3.metric("NEUTRAL", neutral)
    col4.metric("RL BUY", buy)
    col5.metric("RL SELL", sell)


def render_detail():
    st.subheader("Stock Detail")

    predictions = get_latest_predictions()
    tickers = [p["ticker"] for p in predictions] if predictions else WATCHLIST
    selected = st.selectbox("Select Ticker", tickers)

    if not selected:
        return

    # Prediction history
    history = get_prediction_history(selected, limit=100)
    if not history:
        st.info(f"No data for {selected} yet.")
        return

    latest = history[-1]

    # Latest prediction
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Direction", DIRECTION_MAP.get(latest["direction"], "?"))
    col2.metric("Confidence", f"{latest['confidence']:.0%}" if latest["confidence"] else "N/A")
    col3.metric("Agreement", f"{latest['model_agreement']:.0%}" if latest["model_agreement"] else "N/A")
    col4.metric("RL Action", ACTION_NAMES.get(latest["rl_action"], "?"))

    st.metric("Current Price", f"${latest['current_price']:.2f}" if latest["current_price"] else "N/A")

    # Technical data
    tech = latest["technical_data"]
    if tech:
        tcols = st.columns(4)
        tcols[0].metric("RSI", f"{tech.get('rsi', 0):.1f}")
        tcols[1].metric("ADX", f"{tech.get('adx', 0):.1f}")
        tcols[2].metric("MACD Hist", f"{tech.get('macd_hist', 0):.4f}")
        tcols[3].metric("Volatility", f"{tech.get('volatility_20', 0):.2%}")

    # Price history chart
    if len(history) > 1:
        st.markdown("### Prediction History")
        times = [datetime.fromtimestamp(h["timestamp"]) for h in history]
        prices = [h["current_price"] for h in history if h["current_price"]]
        dirs = [h["direction"] for h in history]
        confs = [h["confidence"] for h in history]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4],
                            subplot_titles=["Price & Signals", "Confidence"],
                            vertical_spacing=0.1)

        fig.add_trace(go.Scatter(x=times[:len(prices)], y=prices, name="Price",
                                 line=dict(color="white")), row=1, col=1)

        buy_t = [t for t, d in zip(times, dirs) if d == 1]
        buy_p = [p for p, d in zip(prices, dirs[:len(prices)]) if d == 1]
        sell_t = [t for t, d in zip(times, dirs) if d == -1]
        sell_p = [p for p, d in zip(prices, dirs[:len(prices)]) if d == -1]

        if buy_t:
            fig.add_trace(go.Scatter(x=buy_t, y=buy_p, mode="markers", name="UP",
                                     marker=dict(symbol="triangle-up", size=10, color="lime")),
                          row=1, col=1)
        if sell_t:
            fig.add_trace(go.Scatter(x=sell_t, y=sell_p, mode="markers", name="DOWN",
                                     marker=dict(symbol="triangle-down", size=10, color="red")),
                          row=1, col=1)

        fig.add_trace(go.Bar(x=times, y=confs, name="Confidence",
                             marker_color=["lime" if d == 1 else "red" if d == -1 else "gray"
                                          for d in dirs]),
                      row=2, col=1)

        fig.update_layout(height=500, template="plotly_dark", margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Model metrics
    metrics = get_latest_model_metrics(selected)
    if metrics:
        st.markdown("### Model Metrics")
        col1, col2, col3 = st.columns(3)
        xgb = metrics.get("xgb_metrics", {})
        lstm = metrics.get("lstm_metrics", {})
        trans = metrics.get("trans_metrics", {})

        with col1:
            st.markdown("**XGBoost**")
            st.caption(f"Train Acc: {xgb.get('train_acc', 0):.1%}")
            st.caption(f"Test Acc: {xgb.get('val_acc', 0):.1%}")
        with col2:
            st.markdown("**LSTM**")
            st.caption(f"Val Acc: {lstm.get('best_val_acc', 0):.1%}")
        with col3:
            st.markdown("**Transformer**")
            st.caption(f"Val Acc: {trans.get('best_val_acc', 0):.1%}")

        weights = metrics.get("ensemble_weights", {})
        if weights:
            st.markdown("**Ensemble Weights:** " +
                        " | ".join(f"{k}: {v:.0%}" for k, v in weights.items()))


def render_screener():
    st.subheader("Stock Screener — Pre-computed Results")

    results = get_latest_screening()
    if not results:
        st.info("No screening data yet. Wait for the batch scheduler to run.")
        return

    df = pd.DataFrame(results)
    display_cols = ["ticker", "price", "signal", "composite_score", "risk_score",
                    "reward_ratio", "rsi", "adx", "atr_pct", "volatility", "recommendation"]
    display_cols = [c for c in display_cols if c in df.columns]

    def color_signal(val):
        colors = {"STRONG BUY": "background-color: #1a5c1a", "BUY": "background-color: #2d7d2d",
                  "WATCH": "background-color: #7d7d2d", "AVOID": "background-color: #7d2d2d"}
        return colors.get(val, "")

    st.dataframe(
        df[display_cols].style.map(color_signal, subset=["signal"]),
        use_container_width=True, hide_index=True, height=600,
    )

    # Top picks
    buys = df[df["signal"].isin(["STRONG BUY", "BUY"])]
    if not buys.empty:
        st.markdown("### Top Picks")
        for _, row in buys.head(10).iterrows():
            st.markdown(
                f"**{row['ticker']}** @ ${row['price']:.2f} — "
                f"{row['signal']} | Score: {row['composite_score']:.3f} | "
                f"Risk: {row['risk_score']:.3f} | "
                f"{row.get('recommendation', '')}"
            )


def render_news():
    st.subheader("News & Sentiment")

    predictions = get_latest_predictions()
    tickers = [p["ticker"] for p in predictions] if predictions else WATCHLIST[:20]
    selected = st.selectbox("Select Ticker for Sentiment", tickers, key="news_ticker")

    if not selected:
        return

    sentiment = get_latest_sentiment(selected)
    if not sentiment:
        st.info(f"No sentiment data for {selected} yet.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment", sentiment["overall_sentiment"])
    col2.metric("Confidence", f"{sentiment['confidence']:.0%}")
    col3.metric("News Impact", sentiment["news_impact"])
    col4.metric("Risk Level", sentiment["risk_level"])

    st.info(sentiment["summary"])

    if sentiment["key_drivers"]:
        st.markdown("### Key Drivers")
        for driver in sentiment["key_drivers"]:
            st.markdown(f"- {driver}")


def render_strategy():
    st.subheader("Risk & Strategy")

    predictions = get_latest_predictions()
    tickers = [p["ticker"] for p in predictions] if predictions else WATCHLIST[:20]
    selected = st.selectbox("Select Ticker for Strategy", tickers, key="strategy_ticker")

    if not selected:
        return

    # Risk
    risk = get_latest_risk(selected)
    if risk:
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Risk", risk["overall_risk"])
        col2.metric("Risk Score", f"{risk['risk_score']:.0%}")
        col3.metric("Position Size", risk["position_recommendation"])

        rcols = st.columns(3)
        rcols[0].metric("Volatility Risk", risk["volatility_risk"])
        rcols[1].metric("News Risk", risk["news_risk"])
        rcols[2].metric("Technical Risk", risk["technical_risk"])

        if risk["factors"]:
            st.markdown("#### Risk Factors")
            for f in risk["factors"]:
                st.markdown(f"- {f}")

    # Strategy
    strategy = get_latest_strategy(selected)
    if strategy:
        st.markdown("---")
        st.metric("Recommended Action", strategy["recommended_action"])
        st.caption(f"Confidence: {strategy['confidence']:.0%}")
        st.info(strategy["summary"])

        for w in strategy["warnings"]:
            st.warning(w)

        for s in strategy["strategies"]:
            with st.expander(f"{s['name']} (confidence: {s['confidence']:.0%})"):
                st.write(s["description"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Entry:** " + ", ".join(s["entry_conditions"]))
                    st.markdown(f"**Stop Loss:** {s['stop_loss']}")
                with c2:
                    st.markdown("**Exit:** " + ", ".join(s["exit_conditions"]))
                    st.markdown(f"**Take Profit:** {s['take_profit']}")


def render_system():
    st.subheader("System Status")

    status = get_batch_status()
    if status:
        ts = datetime.fromtimestamp(status["timestamp"])
        st.markdown(f"**Last run:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"**Type:** {status['run_type']}")
        st.markdown(f"**Processed:** {status['tickers_processed']}")
        st.markdown(f"**Failed:** {status['tickers_failed']}")
        st.markdown(f"**Duration:** {status['duration_seconds']:.1f}s")

        if status["errors"]:
            with st.expander(f"Errors ({len(status['errors'])})"):
                for e in status["errors"]:
                    st.code(e)

    st.markdown("---")
    st.markdown("### Commands")
    st.code("# Start scheduler (runs during market hours)\npython -m src.batch.scheduler", language="bash")
    st.code("# Start dashboard\nstreamlit run src/dashboard/batch_app.py", language="bash")
    st.code("# Manual full train\npython -c \"from src.batch.runner import run_full_train; from src.batch.config import WATCHLIST; run_full_train(WATCHLIST)\"", language="bash")


main()

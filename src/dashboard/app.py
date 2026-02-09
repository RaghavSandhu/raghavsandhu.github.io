"""Streamlit dashboard for the stock prediction app."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from src.` imports work everywhere
# (needed for Streamlit Cloud which doesn't set PYTHONPATH)
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.fetcher import fetch_stock_data, get_stock_info
from src.data.features import (
    add_technical_indicators,
    create_targets,
    prepare_dataset,
)
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel
from src.models.transformer_model import TransformerModel
from src.models.ensemble import EnsemblePredictor
from src.models.rl_agent import DQNAgent, TradingEnvironment, ACTION_NAMES
from src.explainer.explainer import PredictionExplainer
from src.backtesting.engine import BacktestEngine, compare_strategies
from src.intelligence.news_fetcher import fetch_all_news
from src.intelligence.llm_analyzer import (
    analyze_sentiment,
    assess_risk,
    has_llm_access,
)
from src.intelligence.strategy_advisor import get_strategy_advice
from src.intelligence.stock_screener import screen_stocks


st.set_page_config(page_title="Stock Trend Predictor", layout="wide")

DIRECTION_MAP = {1: "UP", 0: "NEUTRAL", -1: "DOWN"}


# ── Cached training pipeline (survives tab switches & reruns) ────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_data(ticker, period, interval):
    df = fetch_stock_data(ticker, period=period, interval=interval)
    info = get_stock_info(ticker)
    return df, info


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_news(ticker):
    return fetch_all_news(ticker, max_ticker=8, max_market=3)


@st.cache_resource(ttl=600, show_spinner=False)
def _train_all_models(_df_hash, df_json, ticker, seq_len, horizon, threshold,
                      epochs, rl_episodes, xgb_n):
    """Train all models and return everything needed for display.

    Results are cached by _df_hash + settings so tab switches
    and widget interactions do NOT retrain.
    """
    df = pd.read_json(df_json)

    # Feature engineering
    (
        feature_cols, X_train, X_test,
        y_train_dir, y_test_dir,
        y_train_ret, y_test_ret,
        train_dates, test_dates,
        scaler,
    ) = prepare_dataset(df, sequence_length=seq_len, horizon=horizon, threshold=threshold)

    if len(X_train) < seq_len + 10 or len(X_test) < seq_len + 10:
        return None  # Not enough data

    # XGBoost
    xgb = XGBoostModel(n_estimators=xgb_n)
    xgb_metrics = xgb.train(X_train, y_train_dir, X_test, y_test_dir)

    # LSTM
    lstm = LSTMModel(seq_len=seq_len, epochs=epochs, batch_size=64)
    lstm_metrics = lstm.train(X_train, y_train_dir, X_test, y_test_dir)

    # Transformer
    transformer = TransformerModel(seq_len=seq_len, epochs=epochs, batch_size=64)
    trans_metrics = transformer.train(X_train, y_train_dir, X_test, y_test_dir)

    # Ensemble
    ensemble = EnsemblePredictor(
        {"LSTM": lstm, "XGBoost": xgb, "Transformer": transformer},
        seq_len=seq_len,
    )
    ensemble.update_weights(X_test, y_test_dir)

    # Full df for RL and technical data
    df_full = add_technical_indicators(df)
    df_full = create_targets(df_full, horizon=horizon, threshold=threshold)
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_full.dropna(inplace=True)

    # RL Agent
    test_prices = df_full["Close"].values[-len(X_test):]
    ensemble_preds = ensemble.predict(X_test)
    ensemble_conf = ensemble.get_confidence(X_test)

    min_len = min(len(test_prices), len(ensemble_preds), len(ensemble_conf))
    rl_prices = test_prices[-min_len:]
    rl_preds = ensemble_preds[-min_len:]
    rl_confs = ensemble_conf[-min_len:]
    rl_features = X_test[-min_len:, :8]

    env = TradingEnvironment(rl_prices, rl_preds, rl_confs, rl_features)
    agent = DQNAgent(state_size=env.state_size, epsilon=1.0)
    rl_metrics = agent.train_on_env(env, episodes=rl_episodes)

    # Technical summary
    last_row = df_full.iloc[-1]
    technical_data = {
        "rsi": last_row.get("RSI"),
        "macd_hist": last_row.get("MACD_hist"),
        "adx": last_row.get("ADX"),
        "atr": last_row.get("ATR"),
        "bb_pct": last_row.get("BB_pct"),
        "volatility_20": last_row.get("volatility_20"),
        "volume_ratio": last_row.get("volume_ratio"),
    }
    current_price = float(df_full["Close"].iloc[-1])

    # Pre-compute all predictions and data needed for display
    ens_preds = ensemble.predict(X_test)
    ens_conf = ensemble.get_confidence(X_test)
    agreement = ensemble.get_model_agreement(X_test)
    indiv = ensemble.get_individual_predictions(X_test)

    # RL final action
    last_state = env._get_state()
    rl_action = agent.act(last_state, explore=False)
    q_vals = agent.get_q_values(last_state)

    # SHAP (on subset for speed)
    explainer = PredictionExplainer(xgb, feature_cols)
    shap_subset = X_test[:50] if len(X_test) > 50 else X_test
    importance = explainer.get_global_feature_importance(shap_subset, top_n=15)
    explanation = explainer.explain_prediction(X_test[-1])

    # Backtesting
    engine = BacktestEngine(initial_capital=100_000)
    bt_results = {}
    ens_prices = df_full["Close"].values[-len(X_test):]
    min_bt_len = min(len(ens_prices), len(ens_preds))
    bt_results["Ensemble"] = engine.run(ens_prices[-min_bt_len:], ens_preds[-min_bt_len:])
    xgb_signals = xgb.predict(X_test)
    bt_results["XGBoost Only"] = engine.run(ens_prices[-len(xgb_signals):], xgb_signals)

    env_bt = TradingEnvironment(rl_prices, rl_preds, rl_confs, rl_features)
    state_bt = env_bt.reset()
    rl_signals = []
    done = False
    while not done:
        a = agent.act(state_bt, explore=False)
        rl_signals.append(a - 1)
        state_bt, _, done = env_bt.step(a)
    rl_signals = np.array(rl_signals)
    bt_results["RL Agent"] = engine.run(rl_prices[:len(rl_signals)], rl_signals)
    bh_signals = np.ones(min_bt_len, dtype=int)
    bt_results["Buy & Hold"] = engine.run(ens_prices[-min_bt_len:], bh_signals)
    bt_comparison = compare_strategies(bt_results)

    return {
        "feature_cols": feature_cols,
        "X_train": X_train, "X_test": X_test,
        "y_train_dir": y_train_dir, "y_test_dir": y_test_dir,
        "train_dates": train_dates, "test_dates": test_dates,
        "df_full": df_full,
        "xgb_metrics": xgb_metrics,
        "lstm_metrics": lstm_metrics,
        "trans_metrics": trans_metrics,
        "ensemble_weights": dict(ensemble.weights),
        "rl_metrics": rl_metrics,
        "technical_data": technical_data,
        "current_price": current_price,
        "ens_preds": ens_preds,
        "ens_conf": ens_conf,
        "agreement": agreement,
        "indiv": indiv,
        "rl_action": rl_action,
        "q_vals": q_vals,
        "importance": importance,
        "explanation": explanation,
        "bt_results": bt_results,
        "bt_comparison": bt_comparison,
    }


# ── Rendering functions (pure display, no computation) ───────────────

def render_chart_tab(r):
    st.subheader("Price Chart with Predictions")

    pred_len = len(r["ens_preds"])
    chart_dates = r["test_dates"][-pred_len:]
    chart_prices = r["df_full"]["Close"].values[-len(r["X_test"]):][-pred_len:]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Price & Signals", "Prediction Confidence", "Model Agreement"],
        vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=chart_dates, y=chart_prices, name="Price",
        line=dict(color="white", width=1.5),
    ), row=1, col=1)

    buy_mask = r["ens_preds"] == 1
    sell_mask = r["ens_preds"] == -1
    if buy_mask.any():
        fig.add_trace(go.Scatter(
            x=chart_dates[buy_mask], y=chart_prices[buy_mask],
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
        ), row=1, col=1)
    if sell_mask.any():
        fig.add_trace(go.Scatter(
            x=chart_dates[sell_mask], y=chart_prices[sell_mask],
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=10, color="red"),
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=chart_dates, y=r["ens_conf"], name="Confidence",
        marker_color=np.where(r["ens_preds"] == 1, "lime",
                              np.where(r["ens_preds"] == -1, "red", "gray")),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=chart_dates[-len(r["agreement"]):], y=r["agreement"][-pred_len:],
        name="Agreement", fill="tozeroy", line=dict(color="cyan"),
    ), row=3, col=1)
    fig.update_layout(
        height=700, template="plotly_dark",
        legend=dict(orientation="h", y=1.02), margin=dict(t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Prediction")
    latest_pred = int(r["ens_preds"][-1])
    latest_conf = float(r["ens_conf"][-1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Direction", DIRECTION_MAP[latest_pred])
    col2.metric("Confidence", f"{latest_conf:.1%}")
    col3.metric("Model Agreement", f"{float(r['agreement'][-1]):.0%}")

    st.metric("RL Recommended Action", ACTION_NAMES[r["rl_action"]])
    qv = r["q_vals"]
    st.caption(f"Q-values — SELL: {qv[0]:.3f} | HOLD: {qv[1]:.3f} | BUY: {qv[2]:.3f}")


def render_training_tab(r):
    st.subheader("Model Training Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**XGBoost**")
        st.metric("Train Accuracy", f"{r['xgb_metrics']['train_acc']:.1%}")
        st.metric("Test Accuracy", f"{r['xgb_metrics']['val_acc']:.1%}")
        st.metric("Best Iteration", r["xgb_metrics"]["best_iteration"])
    with col2:
        st.markdown("**LSTM**")
        st.metric("Best Val Accuracy", f"{r['lstm_metrics']['best_val_acc']:.1%}")
        st.metric("Final Train Loss", f"{r['lstm_metrics']['final_train_loss']:.4f}")
        st.metric("Final Val Loss", f"{r['lstm_metrics']['final_val_loss']:.4f}")
    with col3:
        st.markdown("**Transformer**")
        st.metric("Best Val Accuracy", f"{r['trans_metrics']['best_val_acc']:.1%}")
        st.metric("Final Train Loss", f"{r['trans_metrics']['final_train_loss']:.4f}")
        st.metric("Final Val Loss", f"{r['trans_metrics']['final_val_loss']:.4f}")

    st.subheader("Ensemble Weights (dynamically adjusted)")
    weight_df = pd.DataFrame(
        [{"Model": k, "Weight": f"{v:.1%}"} for k, v in r["ensemble_weights"].items()]
    )
    st.dataframe(weight_df, hide_index=True, use_container_width=True)

    st.subheader("Training Curves")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=r["lstm_metrics"]["history"]["train_loss"], name="Train", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(y=r["lstm_metrics"]["history"]["val_loss"], name="Val", line=dict(color="orange")))
        fig.update_layout(title="LSTM Loss", template="plotly_dark", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=r["trans_metrics"]["history"]["train_loss"], name="Train", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(y=r["trans_metrics"]["history"]["val_loss"], name="Val", line=dict(color="orange")))
        fig.update_layout(title="Transformer Loss", template="plotly_dark", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("XGBoost Classification Report")
    report = r["xgb_metrics"]["classification_report"]
    report_df = pd.DataFrame({
        k: v for k, v in report.items()
        if k in ["Down", "Neutral", "Up", "macro avg", "weighted avg"]
    }).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)


def render_explain_tab(r):
    st.subheader("Prediction Explainability")

    st.markdown("### Global Feature Importance (SHAP)")
    importance = r["importance"]
    fig = go.Figure(go.Bar(
        x=[d["importance"] for d in importance],
        y=[d["feature"] for d in importance],
        orientation="h", marker_color="cyan",
    ))
    fig.update_layout(template="plotly_dark", height=400, yaxis=dict(autorange="reversed"), margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latest Prediction Breakdown")
    explanation = r["explanation"]
    top_feats = explanation["top_features"][:10]
    fig = go.Figure(go.Waterfall(
        x=[f["feature"] for f in top_feats],
        y=[f["shap_value"] for f in top_feats],
        connector=dict(line=dict(color="gray")),
        increasing=dict(marker=dict(color="lime")),
        decreasing=dict(marker=dict(color="red")),
    ))
    fig.update_layout(title="Feature Contribution Waterfall", template="plotly_dark", height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Agreement Detail")
    indiv = r["indiv"]
    agree_data = []
    for name, data in indiv.items():
        agree_data.append({
            "Model": name,
            "Latest Prediction": DIRECTION_MAP[int(data["predictions"][-1])],
            "Confidence": f"{float(data['confidence'][-1]):.1%}",
            "Weight": f"{data['weight']:.1%}",
        })
    st.dataframe(pd.DataFrame(agree_data), hide_index=True, use_container_width=True)


def render_rl_tab(r):
    st.subheader("Reinforcement Learning Agent")
    st.markdown("""
    The DQN agent learns optimal **Buy/Hold/Sell** actions through experience replay.
    When predictions go wrong, the agent adjusts its policy — this is the
    **self-correction mechanism**.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Reward (last 20)", f"{r['rl_metrics']['avg_reward']:.2f}")
    col2.metric("Avg Capital (last 20)", f"${r['rl_metrics']['avg_capital']:,.0f}")
    col3.metric("Final Epsilon", f"{r['rl_metrics']['final_epsilon']:.3f}")
    col4.metric("Avg Loss", f"{r['rl_metrics']['avg_loss']:.4f}")

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Episode Rewards", "Capital Curve"])
    fig.add_trace(go.Scatter(y=r["rl_metrics"]["episode_rewards"], name="Reward", line=dict(color="cyan")), row=1, col=1)
    fig.add_trace(go.Scatter(y=r["rl_metrics"]["episode_capitals"], name="Capital", line=dict(color="lime")), row=2, col=1)
    fig.add_hline(y=100_000, line_dash="dash", line_color="gray", annotation_text="Initial Capital", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How does RL self-correction work?"):
        st.markdown("""
        1. **Experience Replay**: Bad trades produce negative rewards; the agent
           samples past experiences to learn from mistakes.
        2. **Double DQN**: A target network prevents overestimation, stabilizing learning.
        3. **Epsilon Decay**: Starts exploring, gradually shifts to exploiting learned policy.
        4. **Dynamic Ensemble Weights**: Down-weights poorly-performing models automatically.
        """)


def render_backtest_tab(r):
    st.subheader("Backtesting Results")

    bt_results = r["bt_results"]
    comparison = r["bt_comparison"]

    fmt_cols = {
        "total_return_pct": "{:.2f}%", "max_drawdown_pct": "{:.2f}%",
        "sharpe_ratio": "{:.2f}", "profit_factor": "{:.2f}",
        "win_rate": "{:.1%}", "avg_win": "{:.4f}", "avg_loss": "{:.4f}",
        "final_capital": "${:,.0f}",
    }
    display_cols = list(fmt_cols.keys()) + ["total_trades"]
    st.dataframe(
        comparison[display_cols].style.format({k: v for k, v in fmt_cols.items() if k in display_cols}),
        use_container_width=True,
    )

    fig = go.Figure()
    for name, result in bt_results.items():
        fig.add_trace(go.Scatter(y=result["equity_curve"], name=name))
    fig.add_hline(y=100_000, line_dash="dash", line_color="gray")
    fig.update_layout(title="Equity Curves", yaxis_title="Capital ($)", template="plotly_dark", height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_news_tab(r, ticker):
    st.subheader("News & Sentiment Analysis")

    if has_llm_access():
        st.success("Claude AI connected — using LLM-powered sentiment analysis")
    else:
        st.info("Using rule-based analysis. Set `ANTHROPIC_API_KEY` for AI-powered insights.")

    news_items = _fetch_news(ticker)
    if not news_items:
        st.warning("No news found. This may be due to network restrictions.")
        return

    sentiment = analyze_sentiment(ticker, news_items, r["current_price"], r["technical_data"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment", sentiment.overall_sentiment)
    col2.metric("Confidence", f"{sentiment.confidence:.0%}")
    col3.metric("News Impact", sentiment.news_impact)
    col4.metric("Risk Level", sentiment.risk_level)

    st.markdown("### Analysis Summary")
    st.info(sentiment.summary)

    st.markdown("### Key Drivers")
    for driver in sentiment.key_drivers:
        st.markdown(f"- {driver}")

    st.markdown("### Latest News")
    for item in news_items[:10]:
        with st.expander(f"[{item.source}] {item.title}"):
            st.write(item.summary[:500])
            st.caption(f"Published: {item.published}")
            if item.url:
                st.caption(f"Source: {item.url}")

    return sentiment


def render_strategy_tab(r, ticker, sentiment):
    st.subheader("Risk Assessment & Strategy Advisor")

    if sentiment is None:
        news_items = _fetch_news(ticker)
        sentiment = analyze_sentiment(ticker, news_items, r["current_price"], r["technical_data"])

    latest_pred = int(r["ens_preds"][-1])
    latest_conf = float(r["ens_conf"][-1])
    model_pred_summary = {
        "direction": DIRECTION_MAP.get(latest_pred, "NEUTRAL"),
        "confidence": f"{latest_conf:.0%}",
        "agreement": f"{float(r['agreement'][-1]):.0%}",
        "rl_action": ACTION_NAMES.get(r["rl_action"], "HOLD"),
    }

    st.markdown("### Risk Assessment")
    risk = assess_risk(ticker, r["current_price"], r["technical_data"], sentiment, model_pred_summary)

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Risk", risk.overall_risk)
    col2.metric("Risk Score", f"{risk.risk_score:.0%}")
    col3.metric("Position Size", risk.position_recommendation)

    risk_cols = st.columns(4)
    risk_cols[0].metric("Volatility Risk", risk.volatility_risk)
    risk_cols[1].metric("News Risk", risk.news_risk)
    risk_cols[2].metric("Technical Risk", risk.technical_risk)
    risk_cols[3].metric("Max Loss Est.", risk.max_loss_estimate)

    st.markdown("#### Risk Factors")
    for factor in risk.factors:
        st.markdown(f"- {factor}")

    if risk.hedging_suggestions:
        st.markdown("#### Hedging Suggestions")
        for s in risk.hedging_suggestions:
            st.markdown(f"- {s}")

    st.markdown("---")
    st.markdown("### Trading Strategies")
    advice = get_strategy_advice(
        ticker, r["current_price"], r["technical_data"], sentiment, risk, model_pred_summary
    )

    st.metric("Recommended Action", advice.recommended_action)
    st.caption(f"Confidence: {advice.confidence:.0%}")
    st.info(advice.summary)

    for w in advice.warnings:
        st.warning(w)

    for strat in advice.strategies:
        with st.expander(f"{strat.name} (confidence: {strat.confidence:.0%})"):
            st.write(strat.description)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Entry Conditions:**")
                for cond in strat.entry_conditions:
                    st.markdown(f"- {cond}")
                st.markdown(f"**Stop Loss:** {strat.stop_loss}")
                st.markdown(f"**Position Size:** {strat.position_size}")
            with c2:
                st.markdown("**Exit Conditions:**")
                for cond in strat.exit_conditions:
                    st.markdown(f"- {cond}")
                st.markdown(f"**Take Profit:** {strat.take_profit}")
                st.markdown(f"**Risk/Reward:** {strat.risk_reward}")
            st.caption(f"Timeframe: {strat.timeframe}")


def render_screener_tab():
    st.subheader("Stock Screener — Low Risk, High Reward")
    st.markdown(
        "Screens stocks for favorable intraday setups based on technical indicators, "
        "volatility, momentum, and volume."
    )

    col1, col2, col3 = st.columns(3)
    max_risk = col1.slider("Max Risk Score", 0.1, 1.0, 0.6, 0.05)
    min_reward = col2.slider("Min Reward Ratio", 1.0, 3.0, 1.3, 0.1)
    screen_interval = col3.selectbox("Screen Interval", ["1d", "60m"], index=0)

    custom_tickers = st.text_input(
        "Custom watchlist (comma-separated, leave empty for default 25 stocks)",
        placeholder="AAPL, MSFT, TSLA, NVDA, ..."
    )

    if st.button("Run Screen", use_container_width=True):
        tickers = None
        if custom_tickers.strip():
            tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

        with st.spinner(f"Screening {len(tickers) if tickers else 25} stocks..."):
            results = screen_stocks(
                tickers=tickers, interval=screen_interval,
                max_risk_score=max_risk, min_reward_ratio=min_reward,
            )

        if results.empty:
            st.warning("No stocks matched the criteria. Try relaxing the filters.")
        else:
            st.success(f"Found {len(results)} opportunities")

            def color_signal(val):
                colors = {
                    "STRONG BUY": "background-color: #1a5c1a",
                    "BUY": "background-color: #2d7d2d",
                    "WATCH": "background-color: #7d7d2d",
                    "AVOID": "background-color: #7d2d2d",
                }
                return colors.get(val, "")

            display = results[[
                "ticker", "price", "signal", "composite_score",
                "risk_score", "reward_ratio", "rsi", "adx",
                "atr_pct", "volatility", "recommendation",
            ]]
            st.dataframe(
                display.style.map(color_signal, subset=["signal"]),
                use_container_width=True, hide_index=True,
            )

            st.markdown("### Top Picks")
            for _, row in results.head(5).iterrows():
                st.markdown(
                    f"**{row['ticker']}** @ ${row['price']} — "
                    f"{row['signal']} | Score: {row['composite_score']:.3f} | "
                    f"Risk: {row['risk_score']:.3f} | R:R {row['reward_ratio']:.1f}x | "
                    f"{row['recommendation']}"
                )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    st.title("Stock Trend Predictor — Intraday Trading")
    st.caption(
        "LSTM + XGBoost + Transformer ensemble | DQN reinforcement learning | "
        "LLM-powered intelligence"
    )

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="AAPL")
        interval = st.selectbox("Interval", ["5m", "15m", "30m", "60m", "1d"], index=0)
        period = st.selectbox("Period", ["7d", "30d", "60d"], index=1)
        horizon = st.slider("Prediction horizon (candles)", 1, 10, 3)
        threshold = st.slider("Direction threshold (%)", 0.05, 0.5, 0.1, 0.05) / 100
        seq_len = st.slider("Sequence length", 10, 50, 20)

        st.subheader("Training")
        speed_mode = st.radio("Speed", ["Fast", "Balanced", "Thorough"],
                              index=0, horizontal=True,
                              help="Fast: ~30s | Balanced: ~1-2min | Thorough: ~3min")

        if speed_mode == "Fast":
            epochs, rl_episodes, xgb_n = 5, 10, 50
        elif speed_mode == "Balanced":
            epochs, rl_episodes, xgb_n = 15, 30, 100
        else:
            epochs, rl_episodes, xgb_n = 30, 50, 200

        st.caption(f"Epochs: {epochs} | RL episodes: {rl_episodes}")

        st.subheader("LLM Intelligence")
        llm_status = "Connected" if has_llm_access() else "Not configured"
        st.caption(f"Claude API: **{llm_status}**")
        if not has_llm_access():
            st.caption("Set `ANTHROPIC_API_KEY` for AI-powered analysis")

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    # Track whether analysis has been run this session
    if run_btn:
        st.session_state["run_key"] = f"{ticker}_{interval}_{period}_{seq_len}_{horizon}_{threshold}_{speed_mode}"

    if "run_key" not in st.session_state:
        st.info("Configure settings in the sidebar and click **Run Analysis** to begin.")
        return

    # ── Fetch data ────────────────────────────────────────────────────
    try:
        df, info = _fetch_data(ticker, period, interval)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return

    st.write(f"**{info['name']}** | {info['sector']} | {len(df)} candles loaded")

    # ── Train (cached — instant on rerun/tab switch) ──────────────────
    df_json = df.to_json()
    df_hash = hash(df_json)  # Cache key

    with st.spinner("Training models... (cached on subsequent runs)"):
        results = _train_all_models(
            df_hash, df_json, ticker, seq_len, horizon, threshold,
            epochs, rl_episodes, xgb_n,
        )

    if results is None:
        st.error("Not enough data for training. Try a longer period or shorter interval.")
        return

    st.write(f"Features: **{len(results['feature_cols'])}** | "
             f"Train: **{len(results['X_train'])}** | "
             f"Test: **{len(results['X_test'])}**")

    # ── Tabs (all display only — no computation) ──────────────────────
    tabs = st.tabs([
        "Live Chart & Predictions", "Training Results", "Explainability",
        "RL Agent", "Backtesting",
        "News & Sentiment", "Risk & Strategy", "Stock Screener",
    ])

    with tabs[0]:
        render_chart_tab(results)
    with tabs[1]:
        render_training_tab(results)
    with tabs[2]:
        render_explain_tab(results)
    with tabs[3]:
        render_rl_tab(results)
    with tabs[4]:
        render_backtest_tab(results)

    sentiment = None
    with tabs[5]:
        sentiment = render_news_tab(results, ticker)
    with tabs[6]:
        render_strategy_tab(results, ticker, sentiment)
    with tabs[7]:
        render_screener_tab()


main()

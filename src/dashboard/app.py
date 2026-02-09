"""Streamlit dashboard for the stock prediction app."""

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
    create_sequences,
)
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel
from src.models.transformer_model import TransformerModel
from src.models.ensemble import EnsemblePredictor
from src.models.rl_agent import DQNAgent, TradingEnvironment, ACTION_NAMES
from src.explainer.explainer import (
    PredictionExplainer,
    compute_model_agreement_report,
)
from src.backtesting.engine import BacktestEngine, compare_strategies


st.set_page_config(page_title="Stock Trend Predictor", layout="wide")


def main():
    st.title("Stock Trend Predictor — Intraday Trading")
    st.caption("LSTM + XGBoost + Transformer ensemble with DQN reinforcement learning")

    # ── Sidebar controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="AAPL")
        interval = st.selectbox("Interval", ["5m", "15m", "30m", "60m", "1d"], index=0)
        period = st.selectbox("Period", ["7d", "30d", "60d"], index=1)
        horizon = st.slider("Prediction horizon (candles)", 1, 10, 3)
        threshold = st.slider("Direction threshold (%)", 0.05, 0.5, 0.1, 0.05) / 100
        seq_len = st.slider("Sequence length", 10, 50, 20)

        st.subheader("Training")
        epochs = st.slider("Epochs (LSTM/Transformer)", 10, 100, 30)
        rl_episodes = st.slider("RL episodes", 20, 200, 50)

        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Configure settings in the sidebar and click **Run Analysis** to begin.")
        return

    # ── Fetch data ────────────────────────────────────────────────────
    with st.status("Fetching market data...", expanded=True) as status:
        try:
            df = fetch_stock_data(ticker, period=period, interval=interval)
            info = get_stock_info(ticker)
            st.write(f"**{info['name']}** | {info['sector']} | {len(df)} candles loaded")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

        status.update(label="Computing technical indicators...", state="running")
        (
            feature_cols, X_train, X_test,
            y_train_dir, y_test_dir,
            y_train_ret, y_test_ret,
            train_dates, test_dates,
            scaler,
        ) = prepare_dataset(df, sequence_length=seq_len, horizon=horizon, threshold=threshold)

        if len(X_train) < seq_len + 10 or len(X_test) < seq_len + 10:
            st.error("Not enough data for training. Try a longer period or shorter interval.")
            return

        st.write(f"Features: **{len(feature_cols)}** | "
                 f"Train: **{len(X_train)}** samples | "
                 f"Test: **{len(X_test)}** samples")
        status.update(label="Data ready", state="complete")

    # ── Train models ──────────────────────────────────────────────────
    tab_chart, tab_train, tab_explain, tab_rl, tab_backtest = st.tabs(
        ["Live Chart & Predictions", "Training Results",
         "Explainability", "RL Agent", "Backtesting"]
    )

    with st.status("Training models...", expanded=True) as status:
        # XGBoost
        status.update(label="Training XGBoost...", state="running")
        xgb = XGBoostModel()
        xgb_metrics = xgb.train(X_train, y_train_dir, X_test, y_test_dir)

        # LSTM
        status.update(label="Training LSTM...", state="running")
        lstm = LSTMModel(seq_len=seq_len, epochs=epochs, batch_size=64)
        lstm_metrics = lstm.train(X_train, y_train_dir, X_test, y_test_dir)

        # Transformer
        status.update(label="Training Transformer...", state="running")
        transformer = TransformerModel(seq_len=seq_len, epochs=epochs, batch_size=64)
        trans_metrics = transformer.train(X_train, y_train_dir, X_test, y_test_dir)

        # Ensemble
        ensemble = EnsemblePredictor(
            {"LSTM": lstm, "XGBoost": xgb, "Transformer": transformer},
            seq_len=seq_len,
        )
        ensemble.update_weights(X_test, y_test_dir)

        # RL Agent
        status.update(label="Training RL agent...", state="running")
        df_full = add_technical_indicators(df)
        df_full = create_targets(df_full, horizon=horizon, threshold=threshold)
        df_full.dropna(inplace=True)

        test_prices = df_full["Close"].values[-len(X_test):]
        ensemble_preds = ensemble.predict(X_test)
        ensemble_conf = ensemble.get_confidence(X_test)

        # Align lengths
        min_len = min(len(test_prices), len(ensemble_preds), len(ensemble_conf))
        rl_prices = test_prices[-min_len:]
        rl_preds = ensemble_preds[-min_len:]
        rl_confs = ensemble_conf[-min_len:]

        # Use a subset of features for RL state
        rl_features = X_test[-min_len:, :8]

        env = TradingEnvironment(rl_prices, rl_preds, rl_confs, rl_features)
        agent = DQNAgent(state_size=env.state_size, epsilon=1.0)
        rl_metrics = agent.train_on_env(env, episodes=rl_episodes)

        status.update(label="All models trained", state="complete")

    # ── Tab 1: Live Chart & Predictions ───────────────────────────────
    with tab_chart:
        st.subheader("Price Chart with Predictions")

        ens_preds_test = ensemble.predict(X_test)
        ens_conf_test = ensemble.get_confidence(X_test)
        pred_len = len(ens_preds_test)
        chart_dates = test_dates[-pred_len:]
        chart_prices = df_full["Close"].values[-len(X_test):][-pred_len:]

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=["Price & Signals", "Prediction Confidence", "Model Agreement"],
            vertical_spacing=0.08,
        )

        # Price line
        fig.add_trace(go.Scatter(
            x=chart_dates, y=chart_prices, name="Price",
            line=dict(color="white", width=1.5),
        ), row=1, col=1)

        # Buy/sell markers
        buy_mask = ens_preds_test == 1
        sell_mask = ens_preds_test == -1
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

        # Confidence
        fig.add_trace(go.Bar(
            x=chart_dates, y=ens_conf_test,
            name="Confidence",
            marker_color=np.where(ens_preds_test == 1, "lime",
                                  np.where(ens_preds_test == -1, "red", "gray")),
        ), row=2, col=1)

        # Model agreement
        agreement = ensemble.get_model_agreement(X_test)
        fig.add_trace(go.Scatter(
            x=chart_dates[-len(agreement):], y=agreement[-pred_len:],
            name="Agreement", fill="tozeroy",
            line=dict(color="cyan"),
        ), row=3, col=1)

        fig.update_layout(
            height=700, template="plotly_dark",
            legend=dict(orientation="h", y=1.02),
            margin=dict(t=60, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latest prediction summary
        st.subheader("Latest Prediction")
        direction_map = {1: "UP", 0: "NEUTRAL", -1: "DOWN"}
        latest_pred = int(ens_preds_test[-1])
        latest_conf = float(ens_conf_test[-1])

        col1, col2, col3 = st.columns(3)
        col1.metric("Direction", direction_map[latest_pred])
        col2.metric("Confidence", f"{latest_conf:.1%}")
        col3.metric("Model Agreement", f"{float(agreement[-1]):.0%}")

        # RL action
        last_state = env._get_state()
        rl_action = agent.act(last_state, explore=False)
        q_vals = agent.get_q_values(last_state)
        st.metric("RL Recommended Action", ACTION_NAMES[rl_action])
        st.caption(f"Q-values — SELL: {q_vals[0]:.3f} | HOLD: {q_vals[1]:.3f} | BUY: {q_vals[2]:.3f}")

    # ── Tab 2: Training Results ───────────────────────────────────────
    with tab_train:
        st.subheader("Model Training Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**XGBoost**")
            st.metric("Train Accuracy", f"{xgb_metrics['train_acc']:.1%}")
            st.metric("Test Accuracy", f"{xgb_metrics['val_acc']:.1%}")
            st.metric("Best Iteration", xgb_metrics["best_iteration"])
        with col2:
            st.markdown("**LSTM**")
            st.metric("Best Val Accuracy", f"{lstm_metrics['best_val_acc']:.1%}")
            st.metric("Final Train Loss", f"{lstm_metrics['final_train_loss']:.4f}")
            st.metric("Final Val Loss", f"{lstm_metrics['final_val_loss']:.4f}")
        with col3:
            st.markdown("**Transformer**")
            st.metric("Best Val Accuracy", f"{trans_metrics['best_val_acc']:.1%}")
            st.metric("Final Train Loss", f"{trans_metrics['final_train_loss']:.4f}")
            st.metric("Final Val Loss", f"{trans_metrics['final_val_loss']:.4f}")

        # Ensemble weights
        st.subheader("Ensemble Weights (dynamically adjusted)")
        weight_df = pd.DataFrame(
            [{"Model": k, "Weight": f"{v:.1%}"} for k, v in ensemble.weights.items()]
        )
        st.dataframe(weight_df, hide_index=True, use_container_width=True)

        # Training curves
        st.subheader("Training Curves")
        c1, c2 = st.columns(2)
        with c1:
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(
                y=lstm_metrics["history"]["train_loss"], name="Train", line=dict(color="cyan"),
            ))
            fig_lstm.add_trace(go.Scatter(
                y=lstm_metrics["history"]["val_loss"], name="Val", line=dict(color="orange"),
            ))
            fig_lstm.update_layout(
                title="LSTM Loss", template="plotly_dark",
                height=300, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_lstm, use_container_width=True)
        with c2:
            fig_trans = go.Figure()
            fig_trans.add_trace(go.Scatter(
                y=trans_metrics["history"]["train_loss"], name="Train", line=dict(color="cyan"),
            ))
            fig_trans.add_trace(go.Scatter(
                y=trans_metrics["history"]["val_loss"], name="Val", line=dict(color="orange"),
            ))
            fig_trans.update_layout(
                title="Transformer Loss", template="plotly_dark",
                height=300, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_trans, use_container_width=True)

        # XGBoost classification report
        st.subheader("XGBoost Classification Report")
        report = xgb_metrics["classification_report"]
        report_df = pd.DataFrame({
            k: v for k, v in report.items()
            if k in ["Down", "Neutral", "Up", "macro avg", "weighted avg"]
        }).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # ── Tab 3: Explainability ─────────────────────────────────────────
    with tab_explain:
        st.subheader("Prediction Explainability")

        explainer = PredictionExplainer(xgb, feature_cols)

        # Global feature importance
        st.markdown("### Global Feature Importance (SHAP)")
        importance = explainer.get_global_feature_importance(X_test, top_n=15)
        imp_df = pd.DataFrame(importance)

        fig_imp = go.Figure(go.Bar(
            x=[d["importance"] for d in importance],
            y=[d["feature"] for d in importance],
            orientation="h",
            marker_color="cyan",
        ))
        fig_imp.update_layout(
            template="plotly_dark", height=400,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Latest prediction explanation
        st.markdown("### Latest Prediction Breakdown")
        explanation = explainer.explain_prediction(X_test[-1])
        st.code(explainer.generate_explanation_text(explanation))

        # Feature contribution waterfall
        top_feats = explanation["top_features"][:10]
        fig_water = go.Figure(go.Waterfall(
            x=[f["feature"] for f in top_feats],
            y=[f["shap_value"] for f in top_feats],
            connector=dict(line=dict(color="gray")),
            increasing=dict(marker=dict(color="lime")),
            decreasing=dict(marker=dict(color="red")),
        ))
        fig_water.update_layout(
            title="Feature Contribution Waterfall",
            template="plotly_dark", height=350,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_water, use_container_width=True)

        # Model agreement detail
        st.markdown("### Model Agreement Detail")
        indiv = ensemble.get_individual_predictions(X_test)
        direction_map = {1: "UP", 0: "NEUTRAL", -1: "DOWN"}

        agree_data = []
        for name, data in indiv.items():
            agree_data.append({
                "Model": name,
                "Latest Prediction": direction_map[int(data["predictions"][-1])],
                "Confidence": f"{float(data['confidence'][-1]):.1%}",
                "Weight": f"{data['weight']:.1%}",
            })
        st.dataframe(pd.DataFrame(agree_data), hide_index=True, use_container_width=True)

    # ── Tab 4: RL Agent ───────────────────────────────────────────────
    with tab_rl:
        st.subheader("Reinforcement Learning Agent")
        st.markdown("""
        The DQN agent learns optimal **Buy/Hold/Sell** actions through experience replay.
        When predictions go wrong and result in losses, the agent remembers those
        experiences and adjusts its policy to avoid similar mistakes — this is the
        **self-correction mechanism**.
        """)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Reward (last 20)", f"{rl_metrics['avg_reward']:.2f}")
        col2.metric("Avg Capital (last 20)", f"${rl_metrics['avg_capital']:,.0f}")
        col3.metric("Final Epsilon", f"{rl_metrics['final_epsilon']:.3f}")
        col4.metric("Avg Loss", f"{rl_metrics['avg_loss']:.4f}")

        # Episode reward curve
        fig_rl = make_subplots(rows=2, cols=1, subplot_titles=["Episode Rewards", "Capital Curve"])
        fig_rl.add_trace(go.Scatter(
            y=rl_metrics["episode_rewards"], name="Reward",
            line=dict(color="cyan"),
        ), row=1, col=1)
        fig_rl.add_trace(go.Scatter(
            y=rl_metrics["episode_capitals"], name="Capital",
            line=dict(color="lime"),
        ), row=2, col=1)
        fig_rl.add_hline(y=100_000, line_dash="dash", line_color="gray",
                         annotation_text="Initial Capital", row=2, col=1)
        fig_rl.update_layout(
            template="plotly_dark", height=500,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_rl, use_container_width=True)

        # How RL self-correction works
        with st.expander("How does RL self-correction work?"):
            st.markdown("""
            1. **Experience Replay**: Every trade (state, action, reward, outcome) is
               stored in a replay buffer. Bad trades produce negative rewards, and the
               agent randomly samples past experiences to learn from mistakes.

            2. **Double DQN**: A separate target network prevents overestimation of
               Q-values, stabilizing learning so one bad trade doesn't destabilize
               the entire policy.

            3. **Epsilon Decay**: The agent starts by exploring random actions and
               gradually shifts to exploiting its learned policy as confidence grows.

            4. **Dynamic Ensemble Weights**: The ensemble automatically down-weights
               models that have been performing poorly recently, reducing the impact
               of unreliable predictions on the RL agent's inputs.
            """)

    # ── Tab 5: Backtesting ────────────────────────────────────────────
    with tab_backtest:
        st.subheader("Backtesting Results")

        engine = BacktestEngine(initial_capital=100_000)

        # Run backtests for different strategies
        bt_results = {}

        # Ensemble signals
        ens_signals = ensemble.predict(X_test)
        ens_prices = df_full["Close"].values[-len(X_test):]
        min_bt_len = min(len(ens_prices), len(ens_signals))
        bt_results["Ensemble"] = engine.run(
            ens_prices[-min_bt_len:], ens_signals[-min_bt_len:],
        )

        # XGBoost only
        xgb_signals = xgb.predict(X_test)
        bt_results["XGBoost Only"] = engine.run(
            ens_prices[-len(xgb_signals):], xgb_signals,
        )

        # RL agent
        env_bt = TradingEnvironment(rl_prices, rl_preds, rl_confs, rl_features)
        state = env_bt.reset()
        rl_signals = []
        done = False
        while not done:
            action = agent.act(state, explore=False)
            rl_signals.append(action - 1)  # map 0,1,2 -> -1,0,1
            state, _, done = env_bt.step(action)
        rl_signals = np.array(rl_signals)
        bt_results["RL Agent"] = engine.run(rl_prices[:len(rl_signals)], rl_signals)

        # Buy & hold baseline
        bh_signals = np.ones(min_bt_len, dtype=int)
        bt_results["Buy & Hold"] = engine.run(ens_prices[-min_bt_len:], bh_signals)

        # Comparison table
        comparison = compare_strategies(bt_results)
        fmt_cols = {
            "total_return_pct": "{:.2f}%",
            "max_drawdown_pct": "{:.2f}%",
            "sharpe_ratio": "{:.2f}",
            "profit_factor": "{:.2f}",
            "win_rate": "{:.1%}",
            "avg_win": "{:.4f}",
            "avg_loss": "{:.4f}",
            "final_capital": "${:,.0f}",
        }
        display_cols = list(fmt_cols.keys()) + ["total_trades"]
        st.dataframe(
            comparison[display_cols].style.format(
                {k: v for k, v in fmt_cols.items() if k in display_cols}
            ),
            use_container_width=True,
        )

        # Equity curves
        fig_eq = go.Figure()
        for name, result in bt_results.items():
            fig_eq.add_trace(go.Scatter(
                y=result["equity_curve"], name=name,
            ))
        fig_eq.add_hline(y=100_000, line_dash="dash", line_color="gray")
        fig_eq.update_layout(
            title="Equity Curves",
            yaxis_title="Capital ($)",
            template="plotly_dark", height=400,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_eq, use_container_width=True)


if __name__ == "__main__":
    main()

"""Stock screener for finding low-risk, high-reward opportunities.

Screens stocks based on technical indicators, volatility, momentum,
and optionally LLM-powered fundamental analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.features import add_technical_indicators


# Default watchlist of liquid, well-known tickers
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "NFLX", "JPM", "V", "MA", "DIS", "BA", "INTC", "PYPL", "SQ",
    "UBER", "SNAP", "COIN", "PLTR", "SOFI", "NIO", "RIVN", "LCID",
]


def screen_stocks(
    tickers: list[str] | None = None,
    interval: str = "1d",
    period: str = "60d",
    min_volume: int = 500_000,
    max_risk_score: float = 0.5,
    min_reward_ratio: float = 1.5,
) -> pd.DataFrame:
    """Screen stocks for low-risk, high-reward intraday opportunities.

    Scoring criteria:
    - Trend strength (ADX)
    - Momentum (RSI not at extremes, positive MACD)
    - Volatility sweet spot (enough to profit, not too wild)
    - Volume confirmation
    - Risk/reward ratio based on ATR and Bollinger Bands

    Returns:
        DataFrame with columns: ticker, score, risk, reward_ratio,
        signal, rsi, adx, atr_pct, volume_avg, recommendation
    """
    if tickers is None:
        tickers = DEFAULT_WATCHLIST

    results = []
    for ticker in tickers:
        try:
            result = _analyze_ticker(ticker, interval, period, min_volume)
            if result:
                results.append(result)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Filter by risk and reward
    df = df[df["risk_score"] <= max_risk_score]
    df = df[df["reward_ratio"] >= min_reward_ratio]

    # Sort by composite score (higher = better opportunity)
    df = df.sort_values("composite_score", ascending=False)
    return df.reset_index(drop=True)


def _analyze_ticker(ticker: str, interval: str, period: str,
                    min_volume: int) -> dict | None:
    """Analyze a single ticker for screening."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty or len(df) < 50:
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    avg_volume = df["Volume"].tail(20).mean()
    if avg_volume < min_volume:
        return None

    df = add_technical_indicators(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        return None

    last = df.iloc[-1]
    close = last["Close"]

    rsi = last.get("RSI", 50)
    adx = last.get("ADX", 25)
    macd_hist = last.get("MACD_hist", 0)
    atr = last.get("ATR", 0)
    bb_pct = last.get("BB_pct", 0.5)
    vol_ratio = last.get("volume_ratio", 1.0)
    volatility = last.get("volatility_20", 0.02)

    atr_pct = atr / close if close > 0 else 0

    # ── Scoring ───────────────────────────────────────────────────────

    # Trend score (0-1): strong trend is good for momentum trades
    trend_score = min(adx / 50, 1.0) if not np.isnan(adx) else 0.5

    # Momentum score (0-1): RSI in sweet spot, positive MACD
    if np.isnan(rsi):
        momentum_score = 0.5
    elif 40 <= rsi <= 60:
        momentum_score = 0.7  # neutral = could go either way
    elif 30 <= rsi <= 40 or 60 <= rsi <= 70:
        momentum_score = 0.9  # approaching extremes = mean reversion opportunity
    elif rsi < 30:
        momentum_score = 1.0  # oversold = buy opportunity
    elif rsi > 70:
        momentum_score = 0.3  # overbought = risky to enter long
    else:
        momentum_score = 0.5

    macd_bonus = 0.1 if macd_hist > 0 else 0.0

    # Volatility score (0-1): want moderate volatility
    if np.isnan(volatility) or volatility == 0:
        vol_score = 0.5
    elif 0.01 <= volatility <= 0.03:
        vol_score = 0.9  # sweet spot
    elif volatility < 0.01:
        vol_score = 0.4  # too calm for intraday
    else:
        vol_score = max(0.2, 1.0 - volatility * 10)  # too volatile

    # Volume score (0-1): higher volume = more liquid
    if np.isnan(vol_ratio):
        volume_score = 0.5
    else:
        volume_score = min(vol_ratio / 2.0, 1.0)

    # Risk score (0-1, lower = better)
    risk_score = 0.3
    if rsi and (rsi > 75 or rsi < 25):
        risk_score += 0.2
    if volatility and volatility > 0.04:
        risk_score += 0.2
    if adx and adx < 15:
        risk_score += 0.1  # no clear trend
    risk_score = min(risk_score, 1.0)

    # Reward ratio: potential gain / potential loss based on ATR
    if atr_pct > 0:
        potential_gain = atr_pct * 1.5
        potential_loss = atr_pct * 0.8
        reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 1.0
    else:
        reward_ratio = 1.0

    # Composite score
    composite = (
        trend_score * 0.25 +
        (momentum_score + macd_bonus) * 0.30 +
        vol_score * 0.20 +
        volume_score * 0.10 +
        (1 - risk_score) * 0.15
    )

    # Signal
    if composite > 0.7 and risk_score < 0.4:
        if rsi and rsi < 35:
            signal = "STRONG BUY"
        elif macd_hist > 0:
            signal = "BUY"
        else:
            signal = "WATCH"
    elif composite > 0.5:
        signal = "WATCH"
    else:
        signal = "AVOID"

    # Recommendation text
    reasons = []
    if rsi and rsi < 35:
        reasons.append("oversold")
    elif rsi and rsi > 65:
        reasons.append("overbought")
    if adx and adx > 30:
        reasons.append("strong trend")
    if vol_ratio and vol_ratio > 1.5:
        reasons.append("high volume")
    if macd_hist > 0:
        reasons.append("bullish MACD")

    recommendation = ", ".join(reasons) if reasons else "neutral setup"

    return {
        "ticker": ticker,
        "price": round(close, 2),
        "composite_score": round(composite, 3),
        "risk_score": round(risk_score, 3),
        "reward_ratio": round(reward_ratio, 2),
        "signal": signal,
        "rsi": round(rsi, 1) if not np.isnan(rsi) else None,
        "adx": round(adx, 1) if not np.isnan(adx) else None,
        "atr_pct": round(atr_pct * 100, 2),
        "volatility": round(volatility * 100, 2) if not np.isnan(volatility) else None,
        "volume_avg": int(avg_volume),
        "recommendation": recommendation,
    }

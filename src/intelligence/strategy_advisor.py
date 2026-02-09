"""LLM-powered strategy advisor for intraday trading.

Combines technical analysis, sentiment, risk assessment, and model
predictions to recommend actionable trading strategies.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from src.intelligence.llm_analyzer import (
    SentimentResult,
    RiskAssessment,
    _call_llm,
    has_llm_access,
)


@dataclass
class TradingStrategy:
    name: str
    description: str
    entry_conditions: list[str]
    exit_conditions: list[str]
    stop_loss: str
    take_profit: str
    position_size: str
    timeframe: str
    risk_reward: str
    confidence: float  # 0-1


@dataclass
class StrategyAdvice:
    recommended_action: str        # BUY / SELL / HOLD / WAIT
    confidence: float
    strategies: list[TradingStrategy]
    market_context: str
    warnings: list[str]
    summary: str


def get_strategy_advice(
    ticker: str,
    current_price: float,
    technical_data: dict,
    sentiment: SentimentResult,
    risk: RiskAssessment,
    model_predictions: dict,
) -> StrategyAdvice:
    """Get comprehensive trading strategy advice.

    Uses LLM when available, falls back to rule-based strategies.
    """
    if has_llm_access():
        return _llm_strategy_advice(
            ticker, current_price, technical_data, sentiment, risk, model_predictions
        )
    return _rule_based_strategy(
        ticker, current_price, technical_data, sentiment, risk, model_predictions
    )


def _llm_strategy_advice(
    ticker, current_price, technical_data, sentiment, risk, model_predictions,
) -> StrategyAdvice:
    """Get strategy advice from Claude."""
    prompt = f"""You are advising an intraday trader on {ticker} at ${current_price:.2f}.

TECHNICAL ANALYSIS:
- RSI: {technical_data.get('rsi', 'N/A')}
- MACD Histogram: {technical_data.get('macd_hist', 'N/A')}
- ADX: {technical_data.get('adx', 'N/A')}
- ATR: {technical_data.get('atr', 'N/A')}
- Bollinger %B: {technical_data.get('bb_pct', 'N/A')}
- 20-day Volatility: {technical_data.get('volatility_20', 'N/A')}
- Volume Ratio: {technical_data.get('volume_ratio', 'N/A')}

SENTIMENT: {sentiment.overall_sentiment} (confidence: {sentiment.confidence:.0%})
Key drivers: {', '.join(sentiment.key_drivers[:3])}

RISK: {risk.overall_risk} (score: {risk.risk_score:.2f})
Position recommendation: {risk.position_recommendation}

ML MODEL PREDICTIONS:
- Direction: {model_predictions.get('direction', 'N/A')}
- Confidence: {model_predictions.get('confidence', 'N/A')}
- Model agreement: {model_predictions.get('agreement', 'N/A')}
- RL action: {model_predictions.get('rl_action', 'N/A')}

Provide 2-3 specific trading strategies optimized for LOW RISK and HIGH REWARD.
Focus on strategies that have clear entry/exit criteria.

Respond in this exact JSON format:
{{
    "recommended_action": "BUY" or "SELL" or "HOLD" or "WAIT",
    "confidence": 0.0 to 1.0,
    "strategies": [
        {{
            "name": "Strategy Name",
            "description": "Brief description",
            "entry_conditions": ["condition1", "condition2"],
            "exit_conditions": ["condition1", "condition2"],
            "stop_loss": "e.g., 1.5% below entry",
            "take_profit": "e.g., 3% above entry",
            "position_size": "e.g., 3% of portfolio",
            "timeframe": "e.g., 15min-1hr",
            "risk_reward": "e.g., 1:2",
            "confidence": 0.0 to 1.0
        }}
    ],
    "market_context": "Brief market overview",
    "warnings": ["warning1", "warning2"],
    "summary": "2-3 sentence actionable summary"
}}

Only return valid JSON."""

    system = (
        "You are a professional quantitative trading strategist. "
        "Provide specific, actionable strategies with exact entry/exit levels. "
        "Always prioritize capital preservation — low risk first, then maximize reward."
    )

    response = _call_llm(prompt, system)
    if response:
        return _parse_strategy_response(response)
    return _rule_based_strategy(
        ticker, current_price, technical_data, sentiment, risk, model_predictions
    )


def _parse_strategy_response(response: str) -> StrategyAdvice:
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)

        strategies = []
        for s in data.get("strategies", []):
            strategies.append(TradingStrategy(
                name=s.get("name", ""),
                description=s.get("description", ""),
                entry_conditions=s.get("entry_conditions", []),
                exit_conditions=s.get("exit_conditions", []),
                stop_loss=s.get("stop_loss", ""),
                take_profit=s.get("take_profit", ""),
                position_size=s.get("position_size", ""),
                timeframe=s.get("timeframe", ""),
                risk_reward=s.get("risk_reward", ""),
                confidence=float(s.get("confidence", 0.5)),
            ))

        return StrategyAdvice(
            recommended_action=data.get("recommended_action", "HOLD"),
            confidence=float(data.get("confidence", 0.5)),
            strategies=strategies,
            market_context=data.get("market_context", ""),
            warnings=data.get("warnings", []),
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return _rule_based_strategy("", 0, {}, SentimentResult(
            "NEUTRAL", 0.5, [], "MEDIUM", "MEDIUM", ""
        ), RiskAssessment(
            "MEDIUM", 0.5, "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM",
            "3%", "2%", [], []
        ), {})


def _rule_based_strategy(
    ticker, current_price, technical_data, sentiment, risk, model_predictions,
) -> StrategyAdvice:
    """Generate strategies using rule-based logic."""
    rsi = technical_data.get("rsi", 50)
    adx = technical_data.get("adx", 25)
    macd_hist = technical_data.get("macd_hist", 0)
    bb_pct = technical_data.get("bb_pct", 0.5)
    atr = technical_data.get("atr", 0)
    vol = technical_data.get("volatility_20", 0.02)

    model_dir = model_predictions.get("direction", "NEUTRAL")
    model_conf = model_predictions.get("confidence", 0.5)

    strategies = []
    warnings = []
    action = "HOLD"
    confidence = 0.5

    # Strategy 1: Mean Reversion (when RSI at extremes + BB at edges)
    if rsi and bb_pct is not None:
        if rsi < 35 and bb_pct < 0.2:
            strategies.append(TradingStrategy(
                name="Mean Reversion Buy",
                description=f"RSI oversold ({rsi:.0f}) and price near lower Bollinger Band — expect bounce",
                entry_conditions=[
                    f"RSI below 35 (currently {rsi:.0f})",
                    f"Price near lower BB (%B = {bb_pct:.2f})",
                    "Wait for first green candle for confirmation",
                ],
                exit_conditions=[
                    "RSI crosses above 50",
                    "Price reaches middle Bollinger Band",
                ],
                stop_loss=f"${current_price * 0.985:.2f} (1.5% below entry)" if current_price else "1.5%",
                take_profit=f"${current_price * 1.03:.2f} (3% above entry)" if current_price else "3%",
                position_size=risk.position_recommendation,
                timeframe="15min - 1hr",
                risk_reward="1:2",
                confidence=0.7,
            ))
            action = "BUY"
            confidence = 0.65

        elif rsi > 65 and bb_pct > 0.8:
            strategies.append(TradingStrategy(
                name="Mean Reversion Sell",
                description=f"RSI overbought ({rsi:.0f}) and price near upper Bollinger Band — expect pullback",
                entry_conditions=[
                    f"RSI above 65 (currently {rsi:.0f})",
                    f"Price near upper BB (%B = {bb_pct:.2f})",
                    "Wait for first red candle for confirmation",
                ],
                exit_conditions=[
                    "RSI crosses below 50",
                    "Price reaches middle Bollinger Band",
                ],
                stop_loss=f"${current_price * 1.015:.2f} (1.5% above entry)" if current_price else "1.5%",
                take_profit=f"${current_price * 0.97:.2f} (3% below entry)" if current_price else "3%",
                position_size=risk.position_recommendation,
                timeframe="15min - 1hr",
                risk_reward="1:2",
                confidence=0.65,
            ))
            action = "SELL"
            confidence = 0.6

    # Strategy 2: Momentum/Trend Following (when ADX strong + MACD confirms)
    if adx and adx > 25:
        if macd_hist and macd_hist > 0 and sentiment.overall_sentiment != "BEARISH":
            strategies.append(TradingStrategy(
                name="Momentum Long",
                description=f"Strong uptrend (ADX={adx:.0f}) with bullish MACD — ride the trend",
                entry_conditions=[
                    f"ADX above 25 (currently {adx:.0f})",
                    "MACD histogram positive",
                    "Enter on pullback to EMA-10",
                ],
                exit_conditions=[
                    "MACD histogram turns negative",
                    "ADX drops below 20",
                    "Trailing stop hit",
                ],
                stop_loss=f"${current_price - atr * 1.5:.2f} (1.5x ATR)" if atr else "1.5x ATR",
                take_profit=f"${current_price + atr * 3:.2f} (3x ATR)" if atr else "3x ATR",
                position_size=risk.position_recommendation,
                timeframe="5min - 30min",
                risk_reward="1:2",
                confidence=0.6,
            ))
            if action == "HOLD":
                action = "BUY"
                confidence = 0.55

        elif macd_hist and macd_hist < 0 and sentiment.overall_sentiment != "BULLISH":
            strategies.append(TradingStrategy(
                name="Momentum Short",
                description=f"Strong downtrend (ADX={adx:.0f}) with bearish MACD — follow the trend",
                entry_conditions=[
                    f"ADX above 25 (currently {adx:.0f})",
                    "MACD histogram negative",
                    "Enter on bounce to EMA-10",
                ],
                exit_conditions=[
                    "MACD histogram turns positive",
                    "ADX drops below 20",
                ],
                stop_loss=f"${current_price + atr * 1.5:.2f} (1.5x ATR)" if atr else "1.5x ATR",
                take_profit=f"${current_price - atr * 3:.2f} (3x ATR)" if atr else "3x ATR",
                position_size=risk.position_recommendation,
                timeframe="5min - 30min",
                risk_reward="1:2",
                confidence=0.55,
            ))
            if action == "HOLD":
                action = "SELL"
                confidence = 0.5

    # Strategy 3: Breakout (low volatility + building volume)
    if vol and vol < 0.015 and adx and adx < 20:
        strategies.append(TradingStrategy(
            name="Volatility Breakout",
            description="Low volatility compression — expect breakout. Wait for direction.",
            entry_conditions=[
                "Price breaks above/below Bollinger Bands",
                "Volume spike > 1.5x average",
                "Enter in direction of breakout",
            ],
            exit_conditions=[
                "Price returns inside Bollinger Bands",
                "Volume drops below average",
            ],
            stop_loss="Opposite Bollinger Band or 1% from entry",
            take_profit="2x the Bollinger Band width",
            position_size=risk.position_recommendation,
            timeframe="5min - 15min",
            risk_reward="1:2.5",
            confidence=0.5,
        ))
        if action == "HOLD":
            action = "WAIT"
            confidence = 0.45

    # If no strategies matched, provide a default
    if not strategies:
        strategies.append(TradingStrategy(
            name="Wait for Setup",
            description="No clear low-risk opportunity detected. Stay patient.",
            entry_conditions=["Wait for RSI to reach extremes (<30 or >70)",
                              "Wait for ADX to rise above 25"],
            exit_conditions=["N/A"],
            stop_loss="N/A",
            take_profit="N/A",
            position_size="0% — do not enter",
            timeframe="N/A",
            risk_reward="N/A",
            confidence=0.3,
        ))
        action = "WAIT"
        confidence = 0.4

    # Warnings
    if risk.overall_risk == "HIGH":
        warnings.append("HIGH RISK environment — reduce position size or stay out")
    if sentiment.news_impact == "HIGH":
        warnings.append("Major news expected — increased volatility likely")
    if model_dir == "NEUTRAL" or model_conf and model_conf < 0.4:
        warnings.append("ML models show low conviction — mixed signals")

    summary = (
        f"{'AI-powered' if has_llm_access() else 'Rule-based'} analysis for {ticker}: "
        f"Recommended action is {action} with {confidence:.0%} confidence. "
        f"{len(strategies)} strategy/strategies identified. "
        f"Risk level: {risk.overall_risk}. "
        f"Set ANTHROPIC_API_KEY for deeper LLM-powered analysis."
        if not has_llm_access() else
        f"Analysis for {ticker}: {action} at {confidence:.0%} confidence."
    )

    return StrategyAdvice(
        recommended_action=action,
        confidence=confidence,
        strategies=strategies,
        market_context=f"Trend: ADX={adx}, Momentum: RSI={rsi}, Vol: {vol}",
        warnings=warnings,
        summary=summary,
    )

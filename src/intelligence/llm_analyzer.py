"""LLM-powered sentiment analysis, news impact, and risk assessment.

Uses the Anthropic Claude API. Requires ANTHROPIC_API_KEY env variable.
Falls back to a rule-based heuristic when no API key is available.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from src.intelligence.news_fetcher import NewsItem, news_to_text

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class SentimentResult:
    overall_sentiment: str       # BULLISH / BEARISH / NEUTRAL
    confidence: float            # 0-1
    key_drivers: list[str]       # Top reasons
    news_impact: str             # HIGH / MEDIUM / LOW
    risk_level: str              # HIGH / MEDIUM / LOW
    summary: str                 # Human-readable summary


@dataclass
class RiskAssessment:
    overall_risk: str            # HIGH / MEDIUM / LOW
    risk_score: float            # 0-1 (1 = highest risk)
    volatility_risk: str
    news_risk: str
    technical_risk: str
    correlation_risk: str
    position_recommendation: str  # % of portfolio
    max_loss_estimate: str
    factors: list[str]
    hedging_suggestions: list[str]


def _get_client() -> "Anthropic | None":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not HAS_ANTHROPIC:
        return None
    return Anthropic(api_key=api_key)


def _call_llm(prompt: str, system: str = "") -> str | None:
    """Call Claude API and return response text. Returns None if unavailable."""
    client = _get_client()
    if client is None:
        return None

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Sentiment Analysis ────────────────────────────────────────────────

def analyze_sentiment(
    ticker: str,
    news_items: list[NewsItem],
    current_price: float,
    technical_summary: dict | None = None,
) -> SentimentResult:
    """Analyze news sentiment for a stock using LLM or rule-based fallback."""
    news_text = news_to_text(news_items)

    tech_context = ""
    if technical_summary:
        tech_context = (
            f"\nTechnical indicators: RSI={technical_summary.get('rsi', 'N/A')}, "
            f"MACD_hist={technical_summary.get('macd_hist', 'N/A')}, "
            f"ADX={technical_summary.get('adx', 'N/A')}, "
            f"BB_pct={technical_summary.get('bb_pct', 'N/A')}"
        )

    prompt = f"""Analyze the following news for {ticker} (current price: ${current_price:.2f}) and provide a trading sentiment assessment.
{tech_context}

NEWS:
{news_text}

Respond in this exact JSON format:
{{
    "overall_sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": 0.0 to 1.0,
    "key_drivers": ["reason1", "reason2", "reason3"],
    "news_impact": "HIGH" or "MEDIUM" or "LOW",
    "risk_level": "HIGH" or "MEDIUM" or "LOW",
    "summary": "2-3 sentence summary of the analysis"
}}

Only return valid JSON, nothing else."""

    system = "You are a senior financial analyst specializing in intraday stock trading. Analyze news sentiment and market impact with precision. Be objective and data-driven."

    llm_response = _call_llm(prompt, system)
    if llm_response:
        return _parse_sentiment_response(llm_response)
    return _rule_based_sentiment(ticker, news_items, technical_summary)


def _parse_sentiment_response(response: str) -> SentimentResult:
    """Parse LLM JSON response into SentimentResult."""
    try:
        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)
        return SentimentResult(
            overall_sentiment=data.get("overall_sentiment", "NEUTRAL"),
            confidence=float(data.get("confidence", 0.5)),
            key_drivers=data.get("key_drivers", []),
            news_impact=data.get("news_impact", "MEDIUM"),
            risk_level=data.get("risk_level", "MEDIUM"),
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, KeyError):
        return SentimentResult(
            overall_sentiment="NEUTRAL", confidence=0.5,
            key_drivers=["Unable to parse LLM response"],
            news_impact="MEDIUM", risk_level="MEDIUM",
            summary=response[:200],
        )


def _rule_based_sentiment(
    ticker: str,
    news_items: list[NewsItem],
    technical_summary: dict | None = None,
) -> SentimentResult:
    """Fallback sentiment analysis using keyword matching."""
    bullish_words = {"surge", "rally", "gain", "rise", "up", "growth", "beat",
                     "strong", "bullish", "record", "high", "profit", "upgrade",
                     "buy", "positive", "outperform", "boost"}
    bearish_words = {"drop", "fall", "decline", "loss", "down", "crash", "weak",
                     "bearish", "low", "miss", "cut", "sell", "negative",
                     "downgrade", "risk", "warning", "concern", "fear"}

    bull_score = 0
    bear_score = 0
    for item in news_items:
        text = f"{item.title} {item.summary}".lower()
        bull_score += sum(1 for w in bullish_words if w in text)
        bear_score += sum(1 for w in bearish_words if w in text)

    total = bull_score + bear_score
    if total == 0:
        sentiment = "NEUTRAL"
        confidence = 0.3
    elif bull_score > bear_score * 1.3:
        sentiment = "BULLISH"
        confidence = min(0.8, bull_score / total)
    elif bear_score > bull_score * 1.3:
        sentiment = "BEARISH"
        confidence = min(0.8, bear_score / total)
    else:
        sentiment = "NEUTRAL"
        confidence = 0.4

    drivers = []
    if bull_score > 0:
        drivers.append(f"Positive news signals ({bull_score} indicators)")
    if bear_score > 0:
        drivers.append(f"Negative news signals ({bear_score} indicators)")
    if technical_summary:
        rsi = technical_summary.get("rsi")
        if rsi and rsi > 70:
            drivers.append("RSI indicates overbought conditions")
        elif rsi and rsi < 30:
            drivers.append("RSI indicates oversold conditions")

    return SentimentResult(
        overall_sentiment=sentiment,
        confidence=confidence,
        key_drivers=drivers or ["Insufficient data for strong signal"],
        news_impact="MEDIUM" if total > 5 else "LOW",
        risk_level="HIGH" if sentiment == "BEARISH" else "MEDIUM",
        summary=(
            f"Rule-based analysis for {ticker}: {bull_score} bullish vs "
            f"{bear_score} bearish signals detected in {len(news_items)} articles. "
            f"Set ANTHROPIC_API_KEY for deeper AI-powered analysis."
        ),
    )


# ── Risk Assessment ───────────────────────────────────────────────────

def assess_risk(
    ticker: str,
    current_price: float,
    technical_data: dict,
    sentiment: SentimentResult,
    model_predictions: dict,
) -> RiskAssessment:
    """Comprehensive risk assessment using LLM or rule-based fallback."""
    prompt = f"""Perform a comprehensive risk assessment for trading {ticker} at ${current_price:.2f}.

TECHNICAL DATA:
- ATR (14-period): {technical_data.get('atr', 'N/A')}
- RSI: {technical_data.get('rsi', 'N/A')}
- ADX: {technical_data.get('adx', 'N/A')}
- Bollinger Band %B: {technical_data.get('bb_pct', 'N/A')}
- 20-day volatility: {technical_data.get('volatility_20', 'N/A')}
- Volume ratio (vs 20-day avg): {technical_data.get('volume_ratio', 'N/A')}

SENTIMENT: {sentiment.overall_sentiment} (confidence: {sentiment.confidence:.0%})
News impact: {sentiment.news_impact}

MODEL PREDICTIONS:
- Ensemble direction: {model_predictions.get('direction', 'N/A')}
- Ensemble confidence: {model_predictions.get('confidence', 'N/A')}
- Model agreement: {model_predictions.get('agreement', 'N/A')}

Respond in this exact JSON format:
{{
    "overall_risk": "HIGH" or "MEDIUM" or "LOW",
    "risk_score": 0.0 to 1.0,
    "volatility_risk": "HIGH/MEDIUM/LOW",
    "news_risk": "HIGH/MEDIUM/LOW",
    "technical_risk": "HIGH/MEDIUM/LOW",
    "correlation_risk": "HIGH/MEDIUM/LOW",
    "position_recommendation": "X% of portfolio",
    "max_loss_estimate": "X% in next session",
    "factors": ["factor1", "factor2", "factor3"],
    "hedging_suggestions": ["suggestion1", "suggestion2"]
}}

Only return valid JSON."""

    system = "You are a quantitative risk analyst. Provide precise, data-driven risk assessments for intraday trading. Be conservative in your recommendations."

    llm_response = _call_llm(prompt, system)
    if llm_response:
        return _parse_risk_response(llm_response)
    return _rule_based_risk(ticker, technical_data, sentiment, model_predictions)


def _parse_risk_response(response: str) -> RiskAssessment:
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)
        return RiskAssessment(
            overall_risk=data.get("overall_risk", "MEDIUM"),
            risk_score=float(data.get("risk_score", 0.5)),
            volatility_risk=data.get("volatility_risk", "MEDIUM"),
            news_risk=data.get("news_risk", "MEDIUM"),
            technical_risk=data.get("technical_risk", "MEDIUM"),
            correlation_risk=data.get("correlation_risk", "MEDIUM"),
            position_recommendation=data.get("position_recommendation", "2-5% of portfolio"),
            max_loss_estimate=data.get("max_loss_estimate", "N/A"),
            factors=data.get("factors", []),
            hedging_suggestions=data.get("hedging_suggestions", []),
        )
    except (json.JSONDecodeError, KeyError):
        return _rule_based_risk("", {}, SentimentResult(
            "NEUTRAL", 0.5, [], "MEDIUM", "MEDIUM", ""
        ), {})


def _rule_based_risk(ticker, technical_data, sentiment, model_predictions):
    """Rule-based risk assessment fallback."""
    risk_score = 0.5
    factors = []

    rsi = technical_data.get("rsi")
    if rsi:
        if rsi > 80 or rsi < 20:
            risk_score += 0.15
            factors.append(f"Extreme RSI ({rsi:.0f}) — high reversal risk")
        elif rsi > 70 or rsi < 30:
            risk_score += 0.05
            factors.append(f"RSI at {rsi:.0f} — approaching extreme")

    adx = technical_data.get("adx")
    if adx and adx > 40:
        factors.append(f"Strong trend (ADX={adx:.0f}) — trend-following favorable")
        risk_score -= 0.05
    elif adx and adx < 20:
        factors.append(f"Weak trend (ADX={adx:.0f}) — choppy, higher risk")
        risk_score += 0.1

    vol = technical_data.get("volatility_20")
    if vol and vol > 0.03:
        risk_score += 0.15
        factors.append(f"High volatility ({vol:.1%}) — increased risk")
    elif vol and vol < 0.01:
        risk_score -= 0.05
        factors.append(f"Low volatility ({vol:.1%}) — calmer market")

    vol_ratio = technical_data.get("volume_ratio")
    if vol_ratio and vol_ratio > 2.0:
        risk_score += 0.1
        factors.append(f"Unusual volume ({vol_ratio:.1f}x avg) — increased activity")

    if sentiment.overall_sentiment == "BEARISH":
        risk_score += 0.1
        factors.append("Bearish news sentiment")
    elif sentiment.overall_sentiment == "BULLISH":
        risk_score -= 0.05

    risk_score = np.clip(risk_score, 0, 1)

    if risk_score > 0.7:
        overall = "HIGH"
        position = "1-2% of portfolio"
        max_loss = f"{risk_score * 5:.1f}% potential drawdown"
    elif risk_score > 0.4:
        overall = "MEDIUM"
        position = "3-5% of portfolio"
        max_loss = f"{risk_score * 3:.1f}% potential drawdown"
    else:
        overall = "LOW"
        position = "5-10% of portfolio"
        max_loss = f"{risk_score * 2:.1f}% potential drawdown"

    hedging = []
    if risk_score > 0.5:
        hedging.append("Use tight stop-loss (1-2% below entry)")
        hedging.append("Consider reducing position size")
    if sentiment.news_impact == "HIGH":
        hedging.append("Wait for news to settle before entering")

    return RiskAssessment(
        overall_risk=overall,
        risk_score=float(risk_score),
        volatility_risk="HIGH" if (vol and vol > 0.03) else "MEDIUM",
        news_risk=sentiment.news_impact,
        technical_risk="HIGH" if (rsi and (rsi > 75 or rsi < 25)) else "MEDIUM",
        correlation_risk="MEDIUM",
        position_recommendation=position,
        max_loss_estimate=max_loss,
        factors=factors or ["Insufficient data — defaulting to medium risk"],
        hedging_suggestions=hedging or ["Standard stop-loss at 2%"],
    )


def has_llm_access() -> bool:
    """Check if LLM (Claude API) is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and HAS_ANTHROPIC

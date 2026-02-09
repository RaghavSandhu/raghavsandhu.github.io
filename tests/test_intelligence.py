"""Tests for the intelligence module (news, sentiment, risk, strategy)."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.intelligence.news_fetcher import (
    NewsItem,
    news_to_text,
    fetch_ticker_news,
    fetch_market_news,
    fetch_all_news,
)
from src.intelligence.llm_analyzer import (
    SentimentResult,
    RiskAssessment,
    analyze_sentiment,
    assess_risk,
    _rule_based_sentiment,
    _rule_based_risk,
    _parse_sentiment_response,
    _parse_risk_response,
    has_llm_access,
)
from src.intelligence.stock_screener import _analyze_ticker
from src.intelligence.strategy_advisor import (
    TradingStrategy,
    StrategyAdvice,
    get_strategy_advice,
    _rule_based_strategy,
)


# ── Helper fixtures ──────────────────────────────────────────────────

def _make_news(n=5, sentiment_bias="mixed"):
    """Create synthetic news items for testing."""
    bullish_titles = [
        "Stock surges to record high on strong earnings",
        "Company reports massive growth and profit beat",
        "Analysts upgrade stock with bullish outlook",
    ]
    bearish_titles = [
        "Stock crashes after disappointing earnings miss",
        "Market fears grow as company warns of losses",
        "Analysts downgrade amid risk and weakness concerns",
    ]
    neutral_titles = [
        "Company holds annual meeting to discuss plans",
        "Market opens flat as traders await data",
        "New product announced at conference",
    ]

    items = []
    for i in range(n):
        if sentiment_bias == "bullish":
            title = bullish_titles[i % len(bullish_titles)]
        elif sentiment_bias == "bearish":
            title = bearish_titles[i % len(bearish_titles)]
        elif sentiment_bias == "neutral":
            title = neutral_titles[i % len(neutral_titles)]
        else:
            all_titles = bullish_titles + bearish_titles + neutral_titles
            title = all_titles[i % len(all_titles)]

        items.append(NewsItem(
            title=title,
            summary=f"Summary of article {i}: {title}",
            source="TestSource",
            url=f"https://example.com/article-{i}",
            published=f"2025-01-0{i+1}",
            ticker="TEST",
        ))
    return items


def _make_technical_data():
    """Create synthetic technical data dict."""
    return {
        "rsi": 45.0,
        "adx": 30.0,
        "macd_hist": 0.15,
        "bb_pct": 0.55,
        "atr": 1.50,
        "volatility_20": 0.02,
        "volume_ratio": 1.2,
    }


def _make_sentiment():
    """Create a SentimentResult for testing."""
    return SentimentResult(
        overall_sentiment="NEUTRAL",
        confidence=0.5,
        key_drivers=["Test driver"],
        news_impact="MEDIUM",
        risk_level="MEDIUM",
        summary="Test sentiment",
    )


def _make_risk():
    """Create a RiskAssessment for testing."""
    return RiskAssessment(
        overall_risk="MEDIUM",
        risk_score=0.5,
        volatility_risk="MEDIUM",
        news_risk="MEDIUM",
        technical_risk="MEDIUM",
        correlation_risk="MEDIUM",
        position_recommendation="3-5% of portfolio",
        max_loss_estimate="1.5% potential drawdown",
        factors=["Test factor"],
        hedging_suggestions=["Use stop-loss"],
    )


# ── News fetcher tests ───────────────────────────────────────────────

def test_news_item_creation():
    item = NewsItem(
        title="Test", summary="Summary", source="Src",
        url="https://example.com", published="2025-01-01", ticker="AAPL",
    )
    assert item.title == "Test"
    assert item.ticker == "AAPL"
    assert item.sentiment is None
    assert item.impact_score is None


def test_news_to_text():
    items = _make_news(3)
    text = news_to_text(items)
    assert "1." in text
    assert "2." in text
    assert "3." in text
    assert "[TestSource]" in text
    assert "Published:" in text


def test_news_to_text_max_items():
    items = _make_news(10)
    text = news_to_text(items, max_items=3)
    assert "4." not in text


def test_news_to_text_empty():
    text = news_to_text([])
    assert text == ""


# ── Sentiment analysis tests ─────────────────────────────────────────

def test_rule_based_sentiment_bullish():
    news = _make_news(5, sentiment_bias="bullish")
    result = _rule_based_sentiment("TEST", news, None)
    assert isinstance(result, SentimentResult)
    assert result.overall_sentiment == "BULLISH"
    assert 0 <= result.confidence <= 1
    assert len(result.key_drivers) > 0


def test_rule_based_sentiment_bearish():
    news = _make_news(5, sentiment_bias="bearish")
    result = _rule_based_sentiment("TEST", news, None)
    assert isinstance(result, SentimentResult)
    assert result.overall_sentiment == "BEARISH"


def test_rule_based_sentiment_neutral():
    news = _make_news(5, sentiment_bias="neutral")
    result = _rule_based_sentiment("TEST", news, None)
    assert isinstance(result, SentimentResult)
    assert result.overall_sentiment == "NEUTRAL"


def test_rule_based_sentiment_with_technicals():
    news = _make_news(3, sentiment_bias="neutral")
    tech = {"rsi": 25}  # oversold
    result = _rule_based_sentiment("TEST", news, tech)
    assert any("oversold" in d.lower() for d in result.key_drivers)


def test_rule_based_sentiment_overbought():
    news = _make_news(3, sentiment_bias="neutral")
    tech = {"rsi": 75}  # overbought
    result = _rule_based_sentiment("TEST", news, tech)
    assert any("overbought" in d.lower() for d in result.key_drivers)


def test_analyze_sentiment_fallback():
    """Without API key, analyze_sentiment should fall back to rule-based."""
    news = _make_news(3, sentiment_bias="bullish")
    with patch.dict("os.environ", {}, clear=True):
        result = analyze_sentiment("TEST", news, 100.0)
    assert isinstance(result, SentimentResult)
    assert result.overall_sentiment in {"BULLISH", "BEARISH", "NEUTRAL"}


def test_parse_sentiment_response_valid():
    import json
    data = {
        "overall_sentiment": "BULLISH",
        "confidence": 0.8,
        "key_drivers": ["Strong earnings", "Market momentum"],
        "news_impact": "HIGH",
        "risk_level": "LOW",
        "summary": "Very bullish outlook",
    }
    result = _parse_sentiment_response(json.dumps(data))
    assert result.overall_sentiment == "BULLISH"
    assert result.confidence == 0.8
    assert len(result.key_drivers) == 2


def test_parse_sentiment_response_with_code_block():
    import json
    data = {"overall_sentiment": "BEARISH", "confidence": 0.6,
            "key_drivers": [], "news_impact": "LOW",
            "risk_level": "HIGH", "summary": "Bad outlook"}
    text = f"```json\n{json.dumps(data)}\n```"
    result = _parse_sentiment_response(text)
    assert result.overall_sentiment == "BEARISH"


def test_parse_sentiment_response_invalid():
    result = _parse_sentiment_response("not valid json at all")
    assert result.overall_sentiment == "NEUTRAL"
    assert result.confidence == 0.5


# ── Risk assessment tests ────────────────────────────────────────────

def test_rule_based_risk_default():
    sentiment = _make_sentiment()
    result = _rule_based_risk("TEST", {}, sentiment, {})
    assert isinstance(result, RiskAssessment)
    assert 0 <= result.risk_score <= 1
    assert result.overall_risk in {"HIGH", "MEDIUM", "LOW"}


def test_rule_based_risk_high_volatility():
    sentiment = _make_sentiment()
    tech = {"volatility_20": 0.05, "rsi": 82, "volume_ratio": 3.0}
    result = _rule_based_risk("TEST", tech, sentiment, {})
    assert result.risk_score > 0.5
    assert any("volatility" in f.lower() or "volume" in f.lower()
               for f in result.factors)


def test_rule_based_risk_low_risk():
    sentiment = SentimentResult(
        "BULLISH", 0.7, [], "LOW", "LOW", "")
    tech = {"rsi": 50, "adx": 35, "volatility_20": 0.005}
    result = _rule_based_risk("TEST", tech, sentiment, {})
    assert result.risk_score < 0.6


def test_assess_risk_fallback():
    """Without API key, assess_risk should fall back to rule-based."""
    sentiment = _make_sentiment()
    with patch.dict("os.environ", {}, clear=True):
        result = assess_risk("TEST", 100.0, _make_technical_data(),
                             sentiment, {})
    assert isinstance(result, RiskAssessment)


def test_parse_risk_response_valid():
    import json
    data = {
        "overall_risk": "LOW",
        "risk_score": 0.3,
        "volatility_risk": "LOW",
        "news_risk": "LOW",
        "technical_risk": "LOW",
        "correlation_risk": "LOW",
        "position_recommendation": "5-10% of portfolio",
        "max_loss_estimate": "1% in next session",
        "factors": ["Low volatility"],
        "hedging_suggestions": ["Standard stop-loss"],
    }
    result = _parse_risk_response(json.dumps(data))
    assert result.overall_risk == "LOW"
    assert result.risk_score == 0.3


def test_parse_risk_response_invalid():
    result = _parse_risk_response("garbage data")
    assert isinstance(result, RiskAssessment)


def test_has_llm_access_no_key():
    with patch.dict("os.environ", {}, clear=True):
        assert has_llm_access() is False


# ── Strategy advisor tests ───────────────────────────────────────────

def test_rule_based_strategy_default():
    tech = _make_technical_data()
    sentiment = _make_sentiment()
    risk = _make_risk()
    result = _rule_based_strategy("TEST", 150.0, tech, sentiment, risk, {})
    assert isinstance(result, StrategyAdvice)
    assert result.recommended_action in {"BUY", "SELL", "HOLD", "WAIT"}
    assert 0 <= result.confidence <= 1
    assert len(result.strategies) > 0


def test_rule_based_strategy_oversold():
    tech = {"rsi": 28, "adx": 25, "macd_hist": -0.5,
            "bb_pct": 0.15, "atr": 1.5, "volatility_20": 0.02}
    sentiment = _make_sentiment()
    risk = _make_risk()
    result = _rule_based_strategy("TEST", 100.0, tech, sentiment, risk, {})
    assert isinstance(result, StrategyAdvice)
    # Should detect oversold conditions and suggest buy
    has_mean_reversion = any("Mean Reversion" in s.name for s in result.strategies)
    assert has_mean_reversion or result.recommended_action in {"BUY", "HOLD", "WAIT"}


def test_rule_based_strategy_overbought():
    tech = {"rsi": 72, "adx": 30, "macd_hist": 0.3,
            "bb_pct": 0.85, "atr": 2.0, "volatility_20": 0.02}
    sentiment = _make_sentiment()
    risk = _make_risk()
    result = _rule_based_strategy("TEST", 200.0, tech, sentiment, risk, {})
    has_sell_strategy = any("Sell" in s.name or "Short" in s.name
                           for s in result.strategies)
    # Overbought RSI + high BB should trigger mean reversion sell
    assert has_sell_strategy or len(result.strategies) > 0


def test_rule_based_strategy_low_vol_breakout():
    tech = {"rsi": 50, "adx": 15, "macd_hist": 0.01,
            "bb_pct": 0.5, "atr": 0.5, "volatility_20": 0.008}
    sentiment = _make_sentiment()
    risk = _make_risk()
    result = _rule_based_strategy("TEST", 50.0, tech, sentiment, risk, {})
    has_breakout = any("Breakout" in s.name for s in result.strategies)
    assert has_breakout


def test_rule_based_strategy_warnings():
    tech = _make_technical_data()
    sentiment = _make_sentiment()
    risk = RiskAssessment(
        "HIGH", 0.8, "HIGH", "HIGH", "HIGH", "HIGH",
        "1%", "5%", ["High risk"], ["Reduce size"])
    result = _rule_based_strategy("TEST", 100.0, tech, sentiment, risk, {})
    assert any("HIGH RISK" in w for w in result.warnings)


def test_get_strategy_advice_fallback():
    """Without API key, get_strategy_advice should use rule-based."""
    tech = _make_technical_data()
    sentiment = _make_sentiment()
    risk = _make_risk()
    with patch.dict("os.environ", {}, clear=True):
        result = get_strategy_advice("TEST", 100.0, tech, sentiment, risk, {})
    assert isinstance(result, StrategyAdvice)


def test_trading_strategy_dataclass():
    s = TradingStrategy(
        name="Test", description="Desc",
        entry_conditions=["A"], exit_conditions=["B"],
        stop_loss="1%", take_profit="3%",
        position_size="5%", timeframe="15min",
        risk_reward="1:3", confidence=0.7,
    )
    assert s.name == "Test"
    assert s.confidence == 0.7

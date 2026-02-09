"""Fetch stock news from yfinance and financial RSS feeds."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import feedparser
import yfinance as yf


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str
    published: str
    ticker: str
    sentiment: str | None = None
    impact_score: float | None = None


# Financial RSS feeds (free, no API key needed)
RSS_FEEDS = {
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
}


def fetch_ticker_news(ticker: str, max_items: int = 10) -> list[NewsItem]:
    """Fetch news for a specific ticker using yfinance."""
    stock = yf.Ticker(ticker)
    items = []

    try:
        news = stock.news or []
    except Exception:
        news = []

    for article in news[:max_items]:
        content = article.get("content", {})
        title = content.get("title", article.get("title", ""))
        summary = content.get("summary", article.get("summary", ""))
        provider = content.get("provider", {})
        source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
        url = content.get("canonicalUrl", {})
        link = url.get("url", article.get("link", "")) if isinstance(url, dict) else article.get("link", "")
        pub_date = content.get("pubDate", article.get("published", ""))

        items.append(NewsItem(
            title=title,
            summary=summary,
            source=source,
            url=link,
            published=pub_date,
            ticker=ticker,
        ))

    return items


def fetch_market_news(max_per_feed: int = 5) -> list[NewsItem]:
    """Fetch general market news from RSS feeds."""
    items = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                items.append(NewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    source=source_name,
                    url=entry.get("link", ""),
                    published=entry.get("published", ""),
                    ticker="MARKET",
                ))
        except Exception:
            continue

    return items


def fetch_all_news(ticker: str, max_ticker: int = 10,
                   max_market: int = 5) -> list[NewsItem]:
    """Fetch both ticker-specific and general market news."""
    ticker_news = fetch_ticker_news(ticker, max_ticker)
    market_news = fetch_market_news(max_market)
    return ticker_news + market_news


def news_to_text(news_items: list[NewsItem], max_items: int = 15) -> str:
    """Convert news items to a formatted text block for LLM analysis."""
    lines = []
    for i, item in enumerate(news_items[:max_items], 1):
        lines.append(
            f"{i}. [{item.source}] {item.title}\n"
            f"   {item.summary[:200]}\n"
            f"   Published: {item.published}"
        )
    return "\n\n".join(lines)

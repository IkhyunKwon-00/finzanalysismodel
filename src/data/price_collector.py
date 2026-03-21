"""Fetch historical stock prices and macro indicators via yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.config import load_config


def fetch_stock_prices(
    symbols: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV for given symbols.

    Returns a MultiIndex DataFrame: index=Date, columns=(metric, symbol).
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
    )
    return data


def fetch_macro_indicators(
    cfg: dict | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily close prices for macro tickers (S&P500, Gold, WTI, BTC, VIX, DXY).

    Returns DataFrame with DatetimeIndex and one column per macro name.
    """
    if cfg is None:
        cfg = load_config()
    macro_cfg = cfg["data"]["macro_tickers"]
    if start_date is None:
        start_date = cfg["data"]["start_date"]
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    tickers = list(macro_cfg.values())
    names = list(macro_cfg.keys())

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=True,
    )

    # Extract Close prices only
    closes = pd.DataFrame(index=raw.index)
    for name, ticker in zip(names, tickers):
        try:
            if len(tickers) == 1:
                closes[f"{name}_close"] = raw["Close"]
            else:
                closes[f"{name}_close"] = raw["Close"][ticker]
        except KeyError:
            closes[f"{name}_close"] = float("nan")

    closes = closes.ffill()
    return closes


def fetch_fundamentals(symbol: str) -> dict:
    """Fetch current fundamental data for a single symbol.

    Returns dict with market_cap, revenue_ttm, operating_margin, etc.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    return {
        "symbol": symbol,
        "market_cap": info.get("marketCap"),
        "revenue_ttm": info.get("totalRevenue"),
        "operating_margin": info.get("operatingMargins"),
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

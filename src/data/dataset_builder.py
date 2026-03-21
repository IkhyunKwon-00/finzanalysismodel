"""Build training dataset: align news articles with price movements and macro context."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.data.external_data import collect_all_news
from src.data.price_collector import (
    fetch_macro_indicators,
    fetch_stock_prices,
)


def compute_future_returns(
    prices: pd.Series,
    date: datetime,
    horizons: list[int],
) -> dict[int, float | None]:
    """Compute future returns at each horizon from a given date.

    Returns {horizon_days: return_pct} or None if data insufficient.
    """
    results = {}
    for h in horizons:
        target_date = date + timedelta(days=h)
        # Find the closest trading day
        mask_current = prices.index <= pd.Timestamp(date)
        mask_future = prices.index <= pd.Timestamp(target_date)
        if mask_current.sum() == 0 or mask_future.sum() == 0:
            results[h] = None
            continue
        current_price = prices[mask_current].iloc[-1]
        future_price = prices[mask_future].iloc[-1]
        if current_price <= 0:
            results[h] = None
            continue
        results[h] = ((future_price - current_price) / current_price) * 100.0
    return results


def return_to_label(return_pct: float | None, threshold: float = 3.0) -> int:
    """Convert return percentage to sentiment label.

    Returns: 0=negative, 1=neutral, 2=positive
    """
    if return_pct is None:
        return 1  # neutral if unknown
    if return_pct >= threshold:
        return 2  # positive
    elif return_pct <= -threshold:
        return 0  # negative
    else:
        return 1  # neutral


def _rolling_features(prices_series: pd.Series, date: pd.Timestamp, windows: list[int]) -> dict:
    """Compute rolling return/volatility features for a price series up to a date."""
    features = {}
    mask = prices_series.index <= date
    available = prices_series[mask]
    if len(available) < 2:
        for w in windows:
            features[f"return_{w}d"] = 0.0
            features[f"vol_{w}d"] = 0.0
        return features

    daily_returns = available.pct_change().dropna()
    for w in windows:
        recent = daily_returns.tail(w)
        features[f"return_{w}d"] = recent.sum() if len(recent) > 0 else 0.0
        features[f"vol_{w}d"] = recent.std() if len(recent) > 1 else 0.0
    return features


def build_dataset(cfg: dict | None = None, output_dir: str = "data/processed") -> Path:
    """Build complete training dataset and save as parquet.

    Steps:
    1. Collect all news (Yahoo Finance + Financial PhraseBank + CSV files)
    2. Fetch stock price history via yfinance
    3. Fetch macro indicators via yfinance
    4. For each news article with a known symbol + date, compute:
       - Future returns at 30/180/360 days → labels
       - Rolling price features at article date
       - Macro features at article date
    5. Save merged dataset as parquet
    """
    if cfg is None:
        cfg = load_config()

    symbols = cfg["data"]["symbols"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]
    horizons = cfg["data"]["horizons"]
    windows = cfg["numerical"]["rolling_windows"]
    raw_dir = Path(cfg["data"].get("raw_dir", "data/raw"))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[1/4] Collecting news from external sources...")
    articles = collect_all_news(symbols=symbols, raw_dir=raw_dir)
    if not articles:
        print("No articles found. Exiting.")
        return out_path / "dataset.parquet"

    print("[2/4] Fetching stock price history...")
    all_prices = fetch_stock_prices(symbols, start_date=start_date, end_date=end_date)

    print("[3/4] Fetching macro indicators...")
    macro_df = fetch_macro_indicators(cfg=cfg, start_date=start_date, end_date=end_date)

    print("[4/4] Building aligned dataset...")
    rows = []
    for article in tqdm(articles, desc="Aligning articles"):
        symbol = article["symbol"].upper()
        pub_date_str = article["published_at"]
        if not pub_date_str or not symbol:
            continue

        try:
            pub_date = pd.Timestamp(pub_date_str).normalize()
        except Exception:
            continue

        # Get price series for this symbol
        try:
            if len(symbols) == 1:
                price_series = all_prices["Close"].dropna()
            else:
                price_series = all_prices[symbol]["Close"].dropna()
        except (KeyError, TypeError):
            continue

        # Future returns → labels
        returns = compute_future_returns(price_series, pub_date, horizons)

        # Rolling stock features
        stock_feats = _rolling_features(price_series, pub_date, windows)

        # Macro features at article date
        macro_feats = {}
        if not macro_df.empty:
            macro_mask = macro_df.index <= pub_date
            if macro_mask.sum() > 0:
                macro_row = macro_df[macro_mask].iloc[-1]
                macro_feats = macro_row.to_dict()

        row = {
            "symbol": symbol,
            "published_at": pub_date_str,
            "title_en": article["title_en"],
            "summary_en": article["summary_en"],
            "topic_id": article["topic_id"],
            "source": article["source"],
        }
        # Add returns and labels
        for h in horizons:
            row[f"return_{h}d"] = returns.get(h)
            row[f"label_{h}d"] = return_to_label(returns.get(h))

        row.update({f"stock_{k}": v for k, v in stock_feats.items()})
        row.update(macro_feats)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill NaN in numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0.0)

    save_path = out_path / "dataset.parquet"
    df.to_parquet(save_path, index=False)
    print(f"Dataset saved: {save_path} ({len(df)} samples)")
    return save_path

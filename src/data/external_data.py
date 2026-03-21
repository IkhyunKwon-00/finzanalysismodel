"""External financial news data sources for model training.

This module is the sole news data provider. Three sources are supported:

1. Yahoo Finance (yfinance)  – per-symbol news, auto-fetched at runtime
2. Financial PhraseBank      – place file at data/raw/financial_phrasebank.txt
3. CSV datasets              – drop any *.csv into data/raw/ for auto-discovery

Column schema (all sources produce the same dict shape):
    title_en      : str   – English headline or sentence
    summary_en    : str   – body text if available, else ""
    symbol        : str   – ticker symbol, "" if not symbol-specific
    topic_id      : str   – source tag (e.g. "symbol:NVDA", "financial_phrasebank")
    published_at  : str   – ISO 8601 datetime string, "" if unknown
    source        : str   – publisher / dataset name
    url           : str   – original URL, "" if not available
    dataset_source: str   – internal tag for bookkeeping
    sentiment_label: str  – pre-assigned label if available, else ""
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ─── Source 1: Yahoo Finance ────────────────────────────────────────────────

def fetch_yahoo_news(
    symbols: list[str],
    max_per_symbol: int = 100,
) -> list[dict]:
    """Fetch recent news from Yahoo Finance for each symbol via yfinance.

    Yahoo Finance typically returns the ~50-100 most recent articles per ticker.
    """
    all_news: list[dict] = []
    for symbol in tqdm(symbols, desc="Yahoo Finance news"):
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news or []
            for item in news_items[:max_per_symbol]:
                pub_ts = item.get("providerPublishTime", 0)
                pub_date = datetime.fromtimestamp(pub_ts).isoformat() if pub_ts else ""
                all_news.append({
                    "title_en": item.get("title", ""),
                    "summary_en": item.get("summary", ""),
                    "symbol": symbol,
                    "topic_id": f"symbol:{symbol}",
                    "published_at": pub_date,
                    "source": item.get("publisher", "Yahoo Finance"),
                    "url": item.get("link", ""),
                    "dataset_source": "yahoo_finance",
                    "sentiment_label": "",
                })
        except Exception as e:
            print(f"  Warning: failed to fetch Yahoo news for {symbol}: {e}")
    print(f"Yahoo Finance: {len(all_news)} articles ({len(symbols)} symbols)")
    return all_news


# ─── Source 2: Financial PhraseBank ─────────────────────────────────────────

def load_financial_phrasebank(filepath: str | Path) -> list[dict]:
    """Load Financial PhraseBank (Malo et al., ~5,000 expert-labeled sentences).

    Download:
        https://huggingface.co/datasets/financial_phrasebank
        → put the .txt file at data/raw/financial_phrasebank.txt

    File format: one entry per line, "sentence@label"
    where label ∈ {positive, neutral, negative}
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Financial PhraseBank not found at {filepath}")
        print("  → Download: https://huggingface.co/datasets/financial_phrasebank")
        return []

    rows: list[dict] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append({
                "title_en": sentence.strip(),
                "summary_en": "",
                "symbol": "",
                "topic_id": "financial_phrasebank",
                "published_at": "",
                "source": "Financial PhraseBank",
                "url": "",
                "dataset_source": "financial_phrasebank",
                "sentiment_label": label.strip(),
            })
    print(f"Financial PhraseBank: {len(rows)} labeled sentences")
    return rows


# ─── Source 3: CSV datasets ──────────────────────────────────────────────────

_CSV_COLUMN_ALIASES = {
    "date": ["date", "Date", "DATE", "published_at", "publishedAt"],
    "text": ["headline", "title", "text", "Headline", "Title", "news"],
    "symbol": ["symbol", "ticker", "Symbol", "Ticker", "SYMBOL"],
    "label": ["label", "sentiment", "Sentiment", "Label"],
}


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def load_csv_dataset(filepath: str | Path) -> list[dict]:
    """Load a single CSV news dataset with auto-detected column names.

    Supported Kaggle datasets (drop *.csv into data/raw/):
    - https://www.kaggle.com/datasets/aaron7sun/stocknews
    - https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
    - Any CSV with headline/title + optional date/symbol/label columns
    """
    filepath = Path(filepath)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Warning: could not load {filepath.name}: {e}")
        return []

    text_col = _resolve_col(df, _CSV_COLUMN_ALIASES["text"])
    if text_col is None:
        print(f"  Skipping {filepath.name}: no recognisable text column")
        return []

    date_col = _resolve_col(df, _CSV_COLUMN_ALIASES["date"])
    symbol_col = _resolve_col(df, _CSV_COLUMN_ALIASES["symbol"])
    label_col = _resolve_col(df, _CSV_COLUMN_ALIASES["label"])

    rows: list[dict] = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or text.lower() in ("nan", ""):
            continue
        rows.append({
            "title_en": text,
            "summary_en": "",
            "symbol": str(row[symbol_col]).strip() if symbol_col else "",
            "topic_id": "external_csv",
            "published_at": str(row[date_col]).strip() if date_col else "",
            "source": filepath.stem,
            "url": "",
            "dataset_source": "csv_import",
            "sentiment_label": str(row[label_col]).strip() if label_col else "",
        })
    print(f"CSV {filepath.name}: {len(rows)} rows")
    return rows


def load_csv_datasets(raw_dir: str | Path) -> list[dict]:
    """Auto-discover and load all *.csv files in raw_dir."""
    raw_dir = Path(raw_dir)
    all_rows: list[dict] = []
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
    for csv_file in csv_files:
        all_rows.extend(load_csv_dataset(csv_file))
    return all_rows


# ─── Unified entry point ─────────────────────────────────────────────────────

def collect_all_news(
    symbols: list[str],
    raw_dir: str | Path = "data/raw",
) -> list[dict]:
    """Collect and deduplicate news from all three sources.

    Sources (in priority order for deduplication):
    1. Yahoo Finance per-symbol news (runtime fetch)
    2. Financial PhraseBank (data/raw/financial_phrasebank.txt)
    3. Any *.csv files in data/raw/

    Returns a deduplicated list of article dicts.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []
    all_articles.extend(fetch_yahoo_news(symbols))
    all_articles.extend(load_financial_phrasebank(raw_dir / "financial_phrasebank.txt"))
    all_articles.extend(load_csv_datasets(raw_dir))

    # Deduplicate by lowercased title
    seen: set[str] = set()
    deduped: list[dict] = []
    for article in all_articles:
        key = article["title_en"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(article)

    print(f"\nTotal news collected: {len(all_articles)} → {len(deduped)} after dedup")
    return deduped

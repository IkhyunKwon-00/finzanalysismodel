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

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ─── Source 1: Yahoo Finance ────────────────────────────────────────────────

def fetch_yahoo_news(
    symbols: list[str],
    max_per_symbol: int = 100,
) -> list[dict]:
    """Fetch recent news from Yahoo Finance for each symbol via yfinance.

    Yahoo Finance typically returns the ~10-50 most recent articles per ticker.
    Handles both legacy flat format and newer nested content format.
    """
    all_news: list[dict] = []
    for symbol in tqdm(symbols, desc="Yahoo Finance news"):
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news or []
            for item in news_items[:max_per_symbol]:
                # Handle nested content format (yfinance >= 0.2.36)
                content = item.get("content", item)
                title = content.get("title", "")
                summary = content.get("summary", content.get("description", ""))
                # Timestamp handling
                pub_date = content.get("pubDate", "")
                if not pub_date:
                    pub_ts = content.get("providerPublishTime", 0)
                    pub_date = datetime.fromtimestamp(pub_ts).isoformat() if pub_ts else ""
                # Provider
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    source = provider.get("displayName", "Yahoo Finance")
                else:
                    source = content.get("publisher", "Yahoo Finance")
                # URL
                canon = content.get("canonicalUrl", {})
                if isinstance(canon, dict):
                    url = canon.get("url", "")
                else:
                    url = content.get("link", "")

                all_news.append({
                    "title_en": title,
                    "summary_en": summary,
                    "symbol": symbol,
                    "topic_id": f"symbol:{symbol}",
                    "published_at": pub_date,
                    "source": source,
                    "url": url,
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
    """Auto-discover and load all *.csv files in raw_dir (recursive, includes Kaggle subdirs)."""
    raw_dir = Path(raw_dir)
    all_rows: list[dict] = []
    csv_files = sorted(raw_dir.glob("**/*.csv"))  # recursive: catches Kaggle downloaded subdirs
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
    for csv_file in csv_files:
        all_rows.extend(load_csv_dataset(csv_file))
    return all_rows


# ─── Source 4: GDELT Historical News ─────────────────────────────────────────

# GDELT Full Text Search API covers online news from 2015-02-19 onward
_GDELT_MIN_DATE = pd.Timestamp("2015-02-19")


def fetch_gdelt_news(
    symbol_queries: dict[str, str],
    start_date: str = "2015-01-01",
    max_per_symbol: int = 2000,
    sleep_between: float = 1.5,
) -> list[dict]:
    """Fetch historical financial news from GDELT 2.0 Full Text Search API.

    GDELT (Global Database of Events, Language and Tone) is a free, global
    news archive covering millions of online articles from ~2015 onward.
    No API key required.

    Args:
        symbol_queries: Mapping of ticker → search query string, e.g.
            {"^GSPC": "S&P 500 stock market", "GC=F": "gold price commodity"}
        start_date: Earliest date to fetch (GDELT covers from 2015-02-19).
        max_per_symbol: Max articles to collect per symbol.
        sleep_between: Seconds between API calls to avoid overloading the service.

    Returns:
        List of article dicts in the standard external_data schema.
    """
    if not symbol_queries:
        return []

    effective_start = max(pd.Timestamp(start_date), _GDELT_MIN_DATE)
    end_ts = pd.Timestamp.now().normalize()

    # Build quarterly date chunks from effective_start to today
    date_chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = effective_start
    while current < end_ts:
        chunk_end = min(current + pd.DateOffset(months=3), end_ts)
        date_chunks.append((current, chunk_end))
        current = chunk_end

    all_articles: list[dict] = []
    for symbol, query in symbol_queries.items():
        symbol_articles: list[dict] = []
        # Distribute evenly; GDELT hard-limit is 250 records per request
        per_chunk = min(max(1, max_per_symbol // max(len(date_chunks), 1)), 250)

        for chunk_start, chunk_end in date_chunks:
            if len(symbol_articles) >= max_per_symbol:
                break
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": per_chunk,
                "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
                "sort": "DateDesc",
                "format": "json",
            }
            url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)
            try:
                with urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                for art in data.get("articles", []):
                    raw_date = art.get("seendate", "")
                    try:
                        pub_dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").replace(
                            tzinfo=timezone.utc
                        )
                        pub_iso = pub_dt.isoformat()
                    except ValueError:
                        pub_iso = raw_date
                    title = art.get("title", "").strip()
                    if not title:
                        continue
                    symbol_articles.append({
                        "title_en": title,
                        "summary_en": "",
                        "symbol": symbol,
                        "topic_id": f"gdelt:{symbol}",
                        "published_at": pub_iso,
                        "source": art.get("domain", "gdelt"),
                        "url": art.get("url", ""),
                        "dataset_source": "gdelt",
                        "sentiment_label": "",
                    })
                time.sleep(sleep_between)
            except Exception as e:
                print(
                    f"  GDELT warning ({symbol} [{chunk_start.date()}–{chunk_end.date()}]): {e}"
                )
                time.sleep(sleep_between * 2)

        kept = symbol_articles[:max_per_symbol]
        all_articles.extend(kept)
        print(f"  GDELT {symbol}: {len(kept)} articles")

    print(f"GDELT: {len(all_articles)} total articles")
    return all_articles


# ─── Source 5: Kaggle Datasets ────────────────────────────────────────────────

_KAGGLE_RECOMMENDED = [
    "aaron7sun/stocknews",              # Dow Jones daily top headlines 2008-2016 (~45 k rows)
    "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",  # ~4 M articles with tickers
]


def download_kaggle_datasets(
    dataset_slugs: list[str] | None = None,
    raw_dir: str | Path = "data/raw",
) -> None:
    """Download Kaggle financial news datasets into raw_dir.

    Downloaded CSV files are automatically picked up by load_csv_datasets().

    Requirements:
        pip install kaggle
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_api_key
        # or create ~/.kaggle/kaggle.json: {"username":"...","key":"..."}

    Trigger from CLI:
        python -m scripts.collect_data --kaggle

    Recommended slugs (defaults if dataset_slugs is None):
        - "aaron7sun/stocknews"            (Dow Jones headlines 2008-2016)
        - "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
    """
    slugs = dataset_slugs if dataset_slugs is not None else _KAGGLE_RECOMMENDED
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # type: ignore[import]
        kaggle.api.authenticate()
    except ImportError:
        print("kaggle package not installed. Run: pip install kaggle")
        print("Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        print("Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or create ~/.kaggle/kaggle.json")
        return

    for slug in slugs:
        try:
            print(f"Downloading Kaggle dataset: {slug} ...")
            kaggle.api.dataset_download_files(
                slug,
                path=str(raw_dir),
                unzip=True,
                quiet=False,
            )
            print(f"  → Extracted to {raw_dir}/")
        except Exception as e:
            print(f"  Failed to download {slug}: {e}")


# ─── Unified entry point ─────────────────────────────────────────────────────

def collect_all_news(
    symbols: list[str],
    raw_dir: str | Path = "data/raw",
    gdelt_queries: dict[str, str] | None = None,
    gdelt_start_date: str = "2015-01-01",
    gdelt_max_per_symbol: int = 2000,
) -> list[dict]:
    """Collect and deduplicate news from all available sources.

    Sources (in priority order for deduplication):
    1. Yahoo Finance per-symbol news (runtime fetch, recent only)
    2. Financial PhraseBank (data/raw/financial_phrasebank.txt)
    3. CSV / Kaggle datasets (data/raw/**/*.csv, recursive)
    4. GDELT Full Text Search (free historical news 2015+, for macro indices)

    Args:
        symbols: Ticker symbols for Yahoo Finance news.
        raw_dir: Directory containing PhraseBank and CSV files.
        gdelt_queries: {symbol: "search query"} for GDELT historical fetch.
            Typically for macro indices, e.g. {"^GSPC": "S&P 500 stock market"}.
        gdelt_start_date: Earliest date for GDELT queries.
        gdelt_max_per_symbol: Max GDELT articles per symbol.

    Returns:
        Deduplicated list of article dicts.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []
    all_articles.extend(fetch_yahoo_news(symbols))
    all_articles.extend(load_financial_phrasebank(raw_dir / "financial_phrasebank.txt"))
    all_articles.extend(load_csv_datasets(raw_dir))
    if gdelt_queries:
        all_articles.extend(
            fetch_gdelt_news(
                gdelt_queries,
                start_date=gdelt_start_date,
                max_per_symbol=gdelt_max_per_symbol,
            )
        )

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

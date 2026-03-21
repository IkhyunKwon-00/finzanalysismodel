"""News collection is handled entirely by external_data.py.

This file is kept for import compatibility but all logic has moved to:
    src/data/external_data.py  →  collect_all_news()
"""

# Re-export the unified entry point for any legacy imports
from src.data.external_data import collect_all_news as fetch_news_articles  # noqa: F401

    return normalized

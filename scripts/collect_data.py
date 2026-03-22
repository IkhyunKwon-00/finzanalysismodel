"""CLI: build training dataset from external news sources + yfinance prices."""

import argparse
from src.config import load_config
from src.data.dataset_builder import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--config", type=str, default=None, help="Override config YAML")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help=(
            "Download Kaggle financial news datasets before building "
            "(requires: pip install kaggle + KAGGLE_USERNAME/KAGGLE_KEY env vars)"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.kaggle:
        from src.data.external_data import download_kaggle_datasets
        kaggle_slugs = cfg["data"].get("kaggle_datasets", [])
        raw_dir = cfg["data"].get("raw_dir", "data/raw")
        download_kaggle_datasets(dataset_slugs=kaggle_slugs, raw_dir=raw_dir)

    build_dataset(cfg=cfg, output_dir=args.output)

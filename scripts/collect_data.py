"""CLI: build training dataset from external news sources + yfinance prices."""

import argparse
from src.config import load_config
from src.data.dataset_builder import build_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--config", type=str, default=None, help="Override config YAML")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_dataset(cfg=cfg, output_dir=args.output)

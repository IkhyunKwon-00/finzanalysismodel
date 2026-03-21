"""PyTorch Dataset that yields (text_tokens, numerical_features, labels) tuples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import load_config


class FinzNewsDataset(Dataset):
    """Dataset for multimodal stock momentum prediction.

    Each sample contains:
    - text: tokenized news title + summary (English)
    - numerical: stock rolling features + macro features (1-D tensor)
    - labels: (label_30d, label_180d, label_360d) as class indices 0/1/2
    - returns: (return_30d, return_180d, return_360d) as float regression targets
    """

    def __init__(
        self,
        parquet_path: str | Path,
        cfg: dict | None = None,
        split: str = "train",
        tokenizer: AutoTokenizer | None = None,
    ):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.max_length = cfg["text_encoder"]["max_length"]
        self.horizons = cfg["data"]["horizons"]

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"])

        # Load dataframe
        df = pd.read_parquet(parquet_path)

        # Split: chronological to avoid look-ahead bias
        df = df.sort_values("published_at").reset_index(drop=True)
        n = len(df)
        val_ratio = cfg["training"]["val_split"]
        test_ratio = cfg["training"]["test_split"]
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        if split == "train":
            df = df.iloc[:train_end]
        elif split == "val":
            df = df.iloc[train_end:val_end]
        elif split == "test":
            df = df.iloc[val_end:]

        self.df = df.reset_index(drop=True)

        # Identify numerical feature columns
        label_cols = [f"label_{h}d" for h in self.horizons]
        return_cols = [f"return_{h}d" for h in self.horizons]
        text_cols = ["title_en", "summary_en", "symbol", "topic_id", "source", "published_at"]
        self.label_cols = label_cols
        self.return_cols = return_cols
        self.num_cols = [
            c for c in df.columns
            if c not in text_cols + label_cols + return_cols
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        # Pre-compute numerical mean/std for normalization (from full df before split)
        full_df = pd.read_parquet(parquet_path)
        train_section = full_df.iloc[:int(len(full_df) * (1 - val_ratio - test_ratio))]
        self.num_mean = train_section[self.num_cols].mean().values.astype(np.float32)
        self.num_std = train_section[self.num_cols].std().values.astype(np.float32)
        self.num_std[self.num_std < 1e-8] = 1.0  # avoid division by zero

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_numerical_features(self) -> int:
        return len(self.num_cols)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Text: concatenate title + summary ────────────────────────
        text = f"{row['title_en']} [SEP] {row['summary_en']}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Numerical features (normalized) ──────────────────────────
        num_vals = row[self.num_cols].values.astype(np.float32)
        num_normalized = (num_vals - self.num_mean) / self.num_std

        # ── Labels & regression targets ──────────────────────────────
        labels = torch.tensor([int(row[c]) for c in self.label_cols], dtype=torch.long)
        returns = torch.tensor(
            [float(row[c]) if pd.notna(row[c]) else 0.0 for c in self.return_cols],
            dtype=torch.float32,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "numerical": torch.tensor(num_normalized, dtype=torch.float32),
            "labels": labels,
            "returns": returns,
        }

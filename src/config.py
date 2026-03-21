"""Loads and merges YAML configuration with environment overrides."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "default.yaml"


def load_config(override_path: str | None = None) -> dict:
    """Load default config, optionally merge with an override YAML, and inject env vars."""
    with open(_DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if override_path and Path(override_path).exists():
        with open(override_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged

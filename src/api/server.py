"""FastAPI inference server – replaces finzfinz's /api/news/sentiment mock endpoint.

Exposes POST /predict endpoint that accepts news text and returns:
  - 30d / 180d / 360d predicted return %
  - 30d / 180d / 360d sentiment label (negative/neutral/positive)
  - confidence scores

This API is designed to be called from finzfinz's Next.js backend.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.config import load_config
from src.models.momentum_model import FinzMomentumModel


# ─── Request / Response schemas ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1, description="News article title (English)")
    summary: str = Field("", description="News article summary/body (English)")
    symbol: str = Field("", description="Stock ticker symbol (e.g., TSLA)")
    numerical_features: list[float] | None = Field(
        None, description="Pre-computed numerical features (optional, for advanced use)"
    )


class HorizonPrediction(BaseModel):
    horizon_days: int
    predicted_return_pct: float
    label: str               # "positive" | "neutral" | "negative"
    confidence_pct: float    # 0-100
    direction: str           # "up" | "down"


class PredictResponse(BaseModel):
    predictions: list[HorizonPrediction]
    model_version: str
    # finzfinz-compatible sentiment (for direct integration)
    sentiment: dict


# ─── Global state ────────────────────────────────────────────────────────────

_state: dict = {}


def _load_model(cfg: dict):
    """Load trained model and tokenizer into memory."""
    model_path = Path(cfg["api"]["model_path"])
    if not model_path.exists():
        print(f"WARNING: Model file not found at {model_path}. Server will return mock predictions.")
        _state["model"] = None
        _state["tokenizer"] = AutoTokenizer.from_pretrained(cfg["text_encoder"]["model_name"])
        _state["cfg"] = cfg
        return

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("config", cfg)
    num_features = checkpoint.get("num_numerical_features", 30)

    model = FinzMomentumModel(
        num_numerical_features=num_features,
        text_model_name=saved_cfg["text_encoder"]["model_name"],
        text_embed_dim=saved_cfg["model"]["text_embed_dim"],
        numerical_hidden=saved_cfg["model"]["numerical_hidden"],
        fusion_hidden=saved_cfg["model"]["fusion_hidden"],
        num_heads=saved_cfg["model"]["num_heads"],
        dropout=0.0,  # no dropout at inference
        num_horizons=saved_cfg["model"]["num_horizons"],
        freeze_text_layers=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(saved_cfg["text_encoder"]["model_name"])

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["cfg"] = saved_cfg
    _state["num_features"] = num_features
    print(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', '?')})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _load_model(cfg)
    yield
    _state.clear()


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Finz Analysis Model API",
    description="Multimodal deep learning model for stock momentum prediction",
    version="0.1.0",
    lifespan=lifespan,
)

cfg = load_config()
origins = cfg.get("api", {}).get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
HORIZONS = [30, 180, 360]


def _mock_prediction() -> list[HorizonPrediction]:
    """Return a reasonable mock prediction when no trained model is available."""
    results = []
    for h in HORIZONS:
        rng = np.random.default_rng()
        ret = float(rng.normal(0, 5))
        direction = "up" if ret >= 0 else "down"
        if abs(ret) >= 3:
            label = "positive" if ret > 0 else "negative"
        else:
            label = "neutral"
        results.append(HorizonPrediction(
            horizon_days=h,
            predicted_return_pct=round(ret, 2),
            label=label,
            confidence_pct=round(float(rng.uniform(60, 95)), 1),
            direction=direction,
        ))
    return results


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict stock momentum from news text."""
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")

    if model is None:
        # No trained model yet → return mock-like predictions
        preds = _mock_prediction()
        best_pred = max(preds, key=lambda p: abs(p.predicted_return_pct))
        return PredictResponse(
            predictions=preds,
            model_version="mock-v1 (no trained model)",
            sentiment={
                "horizonDays": best_pred.horizon_days,
                "expectedMovePct": best_pred.predicted_return_pct,
                "confidencePct": best_pred.confidence_pct,
                "direction": best_pred.direction,
            },
        )

    # ── Tokenize text ────────────────────────────────────────────
    text = f"{req.title} [SEP] {req.summary}"
    encoding = tokenizer(
        text,
        max_length=_state["cfg"]["text_encoder"]["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # ── Numerical features ───────────────────────────────────────
    num_features = _state.get("num_features", 30)
    if req.numerical_features and len(req.numerical_features) == num_features:
        numerical = torch.tensor([req.numerical_features], dtype=torch.float32)
    else:
        numerical = torch.zeros(1, num_features, dtype=torch.float32)

    # ── Inference ────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            numerical=numerical,
        )

    horizons = _state["cfg"]["data"]["horizons"]
    preds = []
    for i, h in enumerate(horizons):
        logits = outputs[f"logits_{i}"][0]
        probs = torch.softmax(logits, dim=-1)
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item() * 100

        return_pct = outputs[f"return_{i}"][0].item()
        label = LABEL_MAP[pred_class]
        direction = "up" if return_pct >= 0 else "down"

        preds.append(HorizonPrediction(
            horizon_days=h,
            predicted_return_pct=round(return_pct, 2),
            label=label,
            confidence_pct=round(confidence, 1),
            direction=direction,
        ))

    # finzfinz-compatible sentiment: use shortest horizon (30d) as primary
    primary = preds[0]
    return PredictResponse(
        predictions=preds,
        model_version=f"finz-momentum-v0.1 (epoch {_state.get('epoch', '?')})",
        sentiment={
            "horizonDays": primary.horizon_days,
            "expectedMovePct": primary.predicted_return_pct,
            "confidencePct": primary.confidence_pct,
            "direction": primary.direction,
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _state.get("model") is not None,
    }

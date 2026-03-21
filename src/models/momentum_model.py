"""Multimodal Momentum Prediction Model.

Architecture:
  ┌──────────────┐   ┌──────────────────┐
  │  News Text   │   │ Numerical Feats  │
  │ (title+body) │   │ (stock + macro)  │
  └──────┬───────┘   └───────┬──────────┘
         │                    │
  ┌──────▼───────┐   ┌───────▼──────────┐
  │  DistilBERT  │   │   MLP Encoder    │
  │  Text Enc.   │   │  (FC → ReLU →    │
  │  → [CLS]     │   │   FC → ReLU)     │
  └──────┬───────┘   └───────┬──────────┘
         │ (768-d)           │ (128-d)
         └─────────┬─────────┘
                   │ concat (896-d)
          ┌────────▼────────┐
          │ Cross-Attention │
          │   Fusion Block  │
          └────────┬────────┘
                   │ (256-d)
          ┌────────▼────────┐
          │  Multi-Horizon  │
          │   Output Heads  │
          │                 │
          │ 30d:  3-class + │
          │       regress   │
          │ 180d: 3-class + │
          │       regress   │
          │ 360d: 3-class + │
          │       regress   │
          └─────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class NumericalEncoder(nn.Module):
    """Encode numerical features into a dense vector."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Fuse text and numerical representations with cross-attention."""

    def __init__(self, text_dim: int, num_dim: int, fusion_dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.num_proj = nn.Linear(num_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_emb: torch.Tensor, num_emb: torch.Tensor) -> torch.Tensor:
        # text_emb: (B, text_dim)  →  (B, 1, fusion_dim)
        # num_emb:  (B, num_dim)   →  (B, 1, fusion_dim)
        t = self.text_proj(text_emb).unsqueeze(1)
        n = self.num_proj(num_emb).unsqueeze(1)

        # Concatenate as key-value sequence: (B, 2, fusion_dim)
        kv = torch.cat([t, n], dim=1)

        # Query from combined representation
        query = (t + n)  # (B, 1, fusion_dim)

        attn_out, _ = self.cross_attn(query, kv, kv)
        attn_out = self.norm(query + attn_out)

        ffn_out = self.ffn(attn_out)
        out = self.final_norm(attn_out + ffn_out)

        return out.squeeze(1)  # (B, fusion_dim)


class HorizonHead(nn.Module):
    """Per-horizon output head: classification (3-class) + regression."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 3),  # negative / neutral / positive
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),  # predicted return %
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(x)    # (B, 3)
        return_pred = self.regressor(x).squeeze(-1)  # (B,)
        return logits, return_pred


class FinzMomentumModel(nn.Module):
    """Multimodal model for stock momentum prediction from news + financial data."""

    def __init__(
        self,
        num_numerical_features: int,
        text_model_name: str = "distilbert-base-uncased",
        text_embed_dim: int = 768,
        numerical_hidden: int = 128,
        fusion_hidden: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_horizons: int = 3,
        freeze_text_layers: int = 4,
    ):
        super().__init__()

        # ── Text Encoder (DistilBERT) ────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_embed_dim = text_embed_dim

        # Freeze early transformer layers for efficiency
        if hasattr(self.text_encoder, "transformer"):
            layers = self.text_encoder.transformer.layer
            for layer in layers[:freeze_text_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # ── Numerical Encoder ────────────────────────────────────────
        self.num_encoder = NumericalEncoder(num_numerical_features, numerical_hidden, dropout)

        # ── Fusion ───────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            text_dim=text_embed_dim,
            num_dim=numerical_hidden,
            fusion_dim=fusion_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── Per-horizon output heads ─────────────────────────────────
        self.horizon_heads = nn.ModuleList([
            HorizonHead(fusion_hidden, dropout) for _ in range(num_horizons)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with keys:
            logits_<i>:  (B, 3) classification logits per horizon
            return_<i>:  (B,)   predicted return % per horizon
        """
        # Text encoding: use [CLS] token representation
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_output.last_hidden_state[:, 0, :]  # [CLS] → (B, 768)

        # Numerical encoding
        num_emb = self.num_encoder(numerical)  # (B, 128)

        # Fusion
        fused = self.fusion(text_emb, num_emb)  # (B, 256)

        # Per-horizon outputs
        outputs = {}
        for i, head in enumerate(self.horizon_heads):
            logits, return_pred = head(fused)
            outputs[f"logits_{i}"] = logits
            outputs[f"return_{i}"] = return_pred

        return outputs

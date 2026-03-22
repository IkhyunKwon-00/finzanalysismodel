"""Sentence-level attribution via Integrated Gradients.

Identifies which sentences in the input text have the highest causal
contribution to the model's classification and regression outputs.

Method: Integrated Gradients (Sundararajan et al., 2017)
- Theoretically grounded (satisfies axioms: sensitivity, implementation invariance)
- Measures actual causal contribution, not just attention correlation
- Works at token level → aggregated to sentence level
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using basic rules."""
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _map_tokens_to_sentences(
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    text: str,
) -> list[int]:
    """Map each token index to a sentence index.

    Returns a list of length seq_len where each element is the sentence index
    that token belongs to. Special tokens get -1.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return [-1] * input_ids.size(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    decoded_chars = []
    for tok in tokens:
        if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "[SEP]"):
            decoded_chars.append(None)
        else:
            # Remove subword prefix (## for BERT-based)
            clean = tok.lstrip("#").lstrip("Ġ")  # handles BERT and GPT-style
            decoded_chars.append(clean.lower())

    # Build sentence mapping by matching tokens to sentence positions
    mapping = [-1] * len(tokens)
    current_sentence = 0
    sentence_text_lower = [s.lower() for s in sentences]
    # Track position within each sentence
    sent_pos = 0
    sent_chars = sentence_text_lower[0] if sentence_text_lower else ""

    for i, tok_text in enumerate(decoded_chars):
        if tok_text is None:
            mapping[i] = -1
            continue

        if current_sentence >= len(sentences):
            mapping[i] = len(sentences) - 1
            continue

        mapping[i] = current_sentence

        # Advance position within sentence
        sent_pos += len(tok_text)
        # Check if we've moved past this sentence
        while current_sentence < len(sentences) - 1 and sent_pos >= len(sent_chars.replace(" ", "")):
            current_sentence += 1
            sent_pos = 0
            if current_sentence < len(sentences):
                sent_chars = sentence_text_lower[current_sentence]
            mapping[i] = current_sentence  # boundary token goes to next sentence

    return mapping


@torch.enable_grad()
def integrated_gradients(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    numerical: torch.Tensor,
    target_horizon: int = 0,
    target_type: str = "classification",
    target_class: int | None = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for input tokens.

    Args:
        model: FinzMomentumModel instance
        input_ids: (1, seq_len) token IDs
        attention_mask: (1, seq_len)
        numerical: (1, num_features)
        target_horizon: which horizon head to attribute (0=30d, 1=180d, 2=360d)
        target_type: "classification" or "regression"
        target_class: if classification, which class to attribute (None = predicted class)
        n_steps: number of interpolation steps (higher = more precise)

    Returns:
        attribution: (seq_len,) token-level attribution scores
    """
    model.eval()

    # Get embedding layer
    embeddings = model.text_encoder.embeddings
    embed_weight = embeddings.word_embeddings

    # Baseline: zero embedding (standard for NLP IG)
    input_embeds = embed_weight(input_ids)  # (1, seq_len, embed_dim)
    baseline_embeds = torch.zeros_like(input_embeds)

    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, n_steps + 1, device=input_ids.device)
    total_grads = torch.zeros_like(input_embeds)

    for alpha in alphas:
        interp_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interp_embeds = interp_embeds.detach().requires_grad_(True)

        # Forward with embeddings directly
        text_output = model.text_encoder(
            inputs_embeds=interp_embeds,
            attention_mask=attention_mask,
        )
        text_emb = text_output.last_hidden_state[:, 0, :]

        num_emb = model.num_encoder(numerical)
        fused = model.fusion(text_emb, num_emb)

        logits, return_pred = model.horizon_heads[target_horizon](fused)

        if target_type == "classification":
            if target_class is None:
                target_class = logits.argmax(dim=-1).item()
            score = logits[0, target_class]
        else:
            score = return_pred[0]

        score.backward(retain_graph=False)
        total_grads += interp_embeds.grad.detach()
        model.zero_grad()

    # Average gradients and multiply by (input - baseline)
    avg_grads = total_grads / (n_steps + 1)
    ig = (input_embeds.detach() - baseline_embeds) * avg_grads

    # Reduce to per-token attribution (L2 norm across embedding dim)
    attribution = ig.squeeze(0).norm(dim=-1)  # (seq_len,)

    return attribution


def get_sentence_attributions(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    numerical: torch.Tensor,
    target_horizon: int = 0,
    target_type: str = "classification",
    target_class: int | None = None,
    n_steps: int = 50,
    top_k: int = 3,
) -> list[dict]:
    """Get top-k sentences ranked by causal contribution to model output.

    Returns list of dicts sorted by attribution score (descending):
        [{"sentence": str, "score": float, "rank": int}, ...]
    """
    attribution = integrated_gradients(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        numerical=numerical,
        target_horizon=target_horizon,
        target_type=target_type,
        target_class=target_class,
        n_steps=n_steps,
    )

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    token_to_sentence = _map_tokens_to_sentences(tokenizer, input_ids.squeeze(0), text)

    # Aggregate token attributions per sentence
    sentence_scores = [0.0] * len(sentences)
    sentence_counts = [0] * len(sentences)
    for tok_idx, sent_idx in enumerate(token_to_sentence):
        if 0 <= sent_idx < len(sentences):
            sentence_scores[sent_idx] += attribution[tok_idx].item()
            sentence_counts[sent_idx] += 1

    # Normalize by token count (average attribution per token in sentence)
    for i in range(len(sentences)):
        if sentence_counts[i] > 0:
            sentence_scores[i] /= sentence_counts[i]

    # Normalize to 0-100 scale
    max_score = max(sentence_scores) if sentence_scores else 1.0
    if max_score > 0:
        sentence_scores = [s / max_score * 100.0 for s in sentence_scores]

    # Sort by score and return top-k
    indexed = [(score, i, sentences[i]) for i, score in enumerate(sentence_scores)]
    indexed.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, idx, sentence) in enumerate(indexed[:top_k], start=1):
        results.append({
            "sentence": sentence,
            "attribution_score": round(score, 2),
            "rank": rank,
        })

    return results

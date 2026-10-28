"""Autoregressive generation with sampling strategies and the KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import GPT, KVCache


@dataclass
class SamplingConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8  # 0 = greedy
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.1
    stop_token: int | None = None
    seed: int | None = None


def apply_repetition_penalty(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style penalty: divide positive logits of already-generated tokens by
    `penalty`, multiply negative ones, so repeats become less likely either way."""
    if penalty == 1.0 or seen.numel() == 0:
        return logits
    logits = logits.clone()
    vals = logits[seen]
    logits[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    kth = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability mass reaches p."""
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum - F.softmax(sorted_logits, dim=-1) >= p  # mass before this token already >= p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf"))
    out[order] = sorted_logits
    return out


def next_token(logits: torch.Tensor, cfg: SamplingConfig, seen: torch.Tensor, gen: torch.Generator | None) -> int:
    logits = apply_repetition_penalty(logits, seen, cfg.repetition_penalty)
    if cfg.temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / cfg.temperature
    if cfg.top_k:
        logits = top_k_filter(logits, cfg.top_k)
    if cfg.top_p:
        logits = top_p_filter(logits, cfg.top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen))


@torch.no_grad()
def generate(model: GPT, prompt: list[int], cfg: SamplingConfig, use_cache: bool = True) -> list[int]:
    """Return prompt + generated tokens. With use_cache=False every step re-runs
    the whole context (the naive baseline used in the benchmark)."""
    model.eval()
    gen = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None
    block = model.cfg.block_size
    tokens = list(prompt) or [0]
    cache = KVCache(model.cfg.n_layer) if use_cache else None
    if use_cache:
        logits, _ = model(torch.tensor([tokens[-block:]]), cache=cache)
    for _ in range(cfg.max_new_tokens):
        if not use_cache:
            logits, _ = model(torch.tensor([tokens[-block:]]))
        t = next_token(logits[0, -1], cfg, torch.tensor(sorted(set(tokens)), dtype=torch.long), gen)
        tokens.append(t)
        if cfg.stop_token is not None and t == cfg.stop_token:
            break
        if use_cache:
            if cache.length >= block:
                # Window full: re-prefill with the most recent half-window.
                cache = KVCache(model.cfg.n_layer)
                logits, _ = model(torch.tensor([tokens[-(block // 2) :]]), cache=cache)
            else:
                logits, _ = model(torch.tensor([[t]]), cache=cache)
    return tokens

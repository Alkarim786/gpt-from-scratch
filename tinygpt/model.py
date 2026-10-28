"""Decoder-only transformer (GPT) with an explicit KV cache.

Each block: x = x + Attn(LN(x)); x = x + MLP(LN(x))  (pre-LayerNorm).
Attention is written out (scores, causal mask, softmax, weighted sum) rather
than calling a fused kernel, so every step is visible and testable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 256  # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True


class KVCache:
    """Per-layer cache of keys and values, shape (B, n_head, T, head_dim)."""

    def __init__(self, n_layer: int) -> None:
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        past = 0
        if cache is not None:
            if cache.k[layer] is not None:
                past = cache.k[layer].shape[2]
                k = torch.cat([cache.k[layer], k], dim=2)
                v = torch.cat([cache.v[layer], v], dim=2)
            cache.k[layer], cache.v[layer] = k, v
        S = k.shape[2]  # total keys = past + T
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, S)
        # Query i (absolute position past+i) may attend to keys 0..past+i.
        q_pos = torch.arange(past, past + T, device=x.device).unsqueeze(1)
        k_pos = torch.arange(S, device=x.device).unsqueeze(0)
        att = att.masked_fill(k_pos > q_pos, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None, layer: int = 0) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache, layer)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        # GPT-2: scale residual projections by 1/sqrt(2 * n_layer).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel() if non_embedding else n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.block_size:
            raise ValueError(f"sequence length {start + T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache, i)
        x = self.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None  # only the last position is needed
        logits = self.lm_head(x)
        # reshape (not view): targets are often a non-contiguous slice like x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.AdamW:
        # Decay matrices (>= 2D) only; never biases, LayerNorm gains or embeddings' 1D params.
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
            betas=betas,
        )

    def config_dict(self) -> dict:
        return asdict(self.cfg)

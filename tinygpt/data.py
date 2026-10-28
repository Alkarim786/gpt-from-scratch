"""Corpus preparation: Wikipedia parquet -> cleaned text -> BPE -> uint16 memmaps."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np

from .bpe import EOT, BPETokenizer

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    # NFC so the same Tamil syllable always has one byte sequence.
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text)
    return _NL.sub("\n\n", text).strip()


def read_articles(parquet: Path, budget_bytes: int, min_chars: int = 200, seed: int = 0) -> list[str]:
    """Shuffle articles deterministically and take them until budget_bytes of UTF-8."""
    import pyarrow.parquet as pq

    texts = pq.read_table(parquet, columns=["text"]).column("text").to_pylist()
    random.Random(seed).shuffle(texts)
    out, used = [], 0
    for t in texts:
        t = clean(t or "")
        if len(t) < min_chars:
            continue
        n = len(t.encode("utf-8"))
        if used + n > budget_bytes:
            break
        out.append(t)
        used += n
    return out


def prepare(
    sources: dict[str, Path],
    out_dir: Path,
    mb_per_lang: float,
    vocab_size: int,
    val_frac: float = 0.05,
    seed: int = 0,
) -> dict:
    """Build tokenizer + train/val token files. Returns a stats dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = int(mb_per_lang * 1e6)
    splits: dict[str, dict[str, list[str]]] = {}
    for lang, path in sources.items():
        arts = read_articles(path, budget, seed=seed)
        n_val = max(1, int(len(arts) * val_frac))
        splits[lang] = {"val": arts[:n_val], "train": arts[n_val:]}

    # Train the tokenizer on the training split only (no val leakage), balanced across languages.
    train_text = [a for lang in splits for a in splits[lang]["train"]]
    tok = BPETokenizer.train(train_text, vocab_size, verbose=True)
    tok.save(out_dir / "tokenizer.json")

    stats: dict = {"vocab_size": tok.vocab_size, "languages": {}}
    for split in ("train", "val"):
        ids: list[int] = []
        docs = [(lang, a) for lang in splits for a in splits[lang][split]]
        random.Random(seed + 1).shuffle(docs)
        for _, a in docs:
            ids.extend(tok.encode(a, allow_special=False))
            ids.append(tok.eot_id)
        np.array(ids, dtype=np.uint16).tofile(out_dir / f"{split}.bin")
        stats[f"{split}_tokens"] = len(ids)
    # Per-language val files for per-language metrics.
    for lang in splits:
        val_text = "\n".join(splits[lang]["val"])
        (out_dir / f"val_{lang}.txt").write_text(val_text, encoding="utf-8")
        ids = tok.encode(val_text, allow_special=False)
        n_bytes = len(val_text.encode("utf-8"))
        stats["languages"][lang] = {
            "articles_train": len(splits[lang]["train"]),
            "articles_val": len(splits[lang]["val"]),
            "val_bytes": n_bytes,
            "val_tokens": len(ids),
            "bytes_per_token": round(n_bytes / max(1, len(ids)), 3),
            "chars_per_token": round(len(val_text) / max(1, len(ids)), 3),
        }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class TokenStream:
    """Random fixed-length windows from a uint16 token file (memory-mapped)."""

    def __init__(self, path: Path, block_size: int, seed: int = 0) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block = block_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} has too few tokens for block_size {block_size}")

    def batch(self, batch_size: int):
        import torch

        ix = self.rng.integers(0, len(self.data) - self.block - 1, size=batch_size)
        x = np.stack([self.data[i : i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.block] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


__all__ = ["EOT", "clean", "prepare", "read_articles", "TokenStream"]

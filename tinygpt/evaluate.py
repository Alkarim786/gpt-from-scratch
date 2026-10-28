"""Evaluate a checkpoint: per-language loss/perplexity/bits-per-byte, KV-cache
speed-up, and samples.

    python -m tinygpt.evaluate --data data/prepared --ckpt runs/base/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

from .bpe import BPETokenizer
from .generate import SamplingConfig, generate
from .model import GPT, GPTConfig


def load(ckpt_path: Path) -> GPT:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def text_metrics(model: GPT, tok: BPETokenizer, text: str, max_tokens: int = 60_000) -> dict:
    """Mean next-token loss over non-overlapping windows, plus bits per byte:
    total nats / ln 2 / UTF-8 bytes covered. BPB is comparable across tokenizers;
    perplexity per token is not."""
    ids = tok.encode(text, allow_special=False)[:max_tokens]
    block = model.cfg.block_size
    total_nats, n_tok = 0.0, 0
    for i in range(0, len(ids) - 1, block):
        chunk = ids[i : i + block + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]])
        y = torch.tensor([chunk[1:]])
        _, loss = model(x, y)
        total_nats += loss.item() * y.numel()
        n_tok += y.numel()
    covered_bytes = sum(len(tok.token_bytes(t)) for t in ids[1 : n_tok + 1])
    loss = total_nats / n_tok
    return {
        "tokens": n_tok,
        "loss": round(loss, 4),
        "perplexity": round(math.exp(loss), 2),
        "bits_per_byte": round(total_nats / math.log(2) / covered_bytes, 4),
    }


def bench_kv_cache(model: GPT, tok: BPETokenizer, prompt: str, new_tokens: int, repeats: int = 3) -> dict:
    ids = tok.encode(prompt)
    cfg = SamplingConfig(max_new_tokens=new_tokens, temperature=0, top_k=None, top_p=None, repetition_penalty=1.0)
    out = {}
    for use_cache in (False, True):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = generate(model, ids, cfg, use_cache=use_cache)
            best = min(best, time.perf_counter() - t0)
        out["with_cache" if use_cache else "no_cache"] = {
            "seconds": round(best, 3),
            "tokens_per_s": round(new_tokens / best, 1),
        }
        out.setdefault("outputs", []).append(res)
    same = out.pop("outputs")
    out["identical_greedy_output"] = same[0] == same[1]
    out["speedup"] = round(out["no_cache"]["seconds"] / out["with_cache"]["seconds"], 2)
    out["prompt_tokens"] = len(ids)
    out["new_tokens"] = new_tokens
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/base/ckpt.pt"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tok = BPETokenizer.load(args.data / "tokenizer.json")
    model = load(args.ckpt)
    res: dict = {"cpu": platform.processor(), "threads": torch.get_num_threads(), "languages": {}}
    for f in sorted(args.data.glob("val_*.txt")):
        lang = f.stem.removeprefix("val_")
        res["languages"][lang] = text_metrics(model, tok, f.read_text(encoding="utf-8"))
    res["kv_cache"] = bench_kv_cache(model, tok, "The history of", new_tokens=200)
    samples = {}
    for name, prompt in {"en": "The river", "ta": "தமிழ்நாடு"}.items():
        ids = generate(model, tok.encode(prompt), SamplingConfig(max_new_tokens=60, temperature=0.8, seed=7))
        samples[name] = tok.decode(ids)
    res["samples"] = samples
    out = args.out or args.ckpt.parent / "eval.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    # Windows consoles default to cp1252, which can't print Tamil.
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

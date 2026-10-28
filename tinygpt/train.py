"""Training loop: AdamW, linear warmup + cosine decay, grad clipping, eval, checkpoints.

python -m tinygpt.train --data data/prepared --out runs/base --steps 1500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .bpe import BPETokenizer
from .data import TokenStream
from .model import GPT, GPTConfig


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model: GPT, stream: TokenStream, batches: int, batch_size: int) -> float:
    model.eval()
    losses = [model(*stream.batch(batch_size))[1].item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/prepared"))
    ap.add_argument("--out", type=Path, default=Path("runs/base"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true", help="continue from OUT/ckpt.pt (same --steps schedule)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(args.data / "tokenizer.json")
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(args.lr, args.weight_decay)
    start = 0
    if args.resume:
        ck = torch.load(args.out / "ckpt.pt", map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        else:
            print("note: checkpoint has no optimizer state; AdamW moments restart from zero")
        start = ck["step"]
        print(f"resumed from step {start}")
    # Offset the data seed on resume so we don't replay the same batches.
    train = TokenStream(args.data / "train.bin", args.block, seed=args.seed + start)
    val = TokenStream(args.data / "val.bin", args.block, seed=args.seed + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"params: {model.num_params() / 1e6:.2f}M non-embedding, {model.num_params(False) / 1e6:.2f}M total; device={device}"
    )

    log = []
    t0 = time.time()
    tokens_seen = 0
    for step in range(start, args.steps + 1):
        if (step % args.eval_every == 0 or step == args.steps) and not (args.resume and step == start):
            vl = estimate_loss(model, val, args.eval_batches, args.batch)
            tl = estimate_loss(model, train, args.eval_batches, args.batch)
            entry = {
                "step": step,
                "train_loss": round(tl, 4),
                "val_loss": round(vl, 4),
                "val_ppl": round(math.exp(vl), 2),
                "elapsed_s": round(time.time() - t0, 1),
                "tokens_seen": tokens_seen,
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)
            torch.save(
                {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg), "step": step},
                args.out / "ckpt.pt",
            )
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        x, y = train.batch(args.batch)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += x.numel()

    train_s = time.time() - t0
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resumed_from_step": start,
        "config": asdict(cfg),
        "params_total": model.num_params(False),
        "params_non_embedding": model.num_params(),
        "train_seconds": round(train_s, 1),
        "tokens_per_second": round(tokens_seen / train_s, 1),
        "device": device,
        "threads": torch.get_num_threads(),
        "log": log,
    }
    (args.out / "train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done in {train_s:.0f}s ({summary['tokens_per_second']:.0f} tok/s)")


if __name__ == "__main__":
    main()

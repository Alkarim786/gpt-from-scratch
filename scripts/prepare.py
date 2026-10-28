"""Build the BPE tokenizer and token files from the raw parquet shards.

python scripts/prepare.py --raw data/raw --out data/prepared --mb-per-lang 15 --vocab 8192
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinygpt.data import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/prepared"))
    ap.add_argument("--mb-per-lang", type=float, default=15)
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()
    t0 = time.time()
    stats = prepare(
        {"ta": args.raw / "ta.parquet", "en": args.raw / "en.parquet"}, args.out, args.mb_per_lang, args.vocab
    )
    stats["prepare_seconds"] = round(time.time() - t0, 1)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

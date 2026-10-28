"""Download one parquet shard each of Tamil Wikipedia and Simple English
Wikipedia (wikimedia/wikipedia, 2023-11-01 dump, CC BY-SA 4.0) from the
Hugging Face hub.  ~300 MB total; only a few MB of text per language is used.

    python scripts/download_data.py --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

SHARDS = {
    "ta": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.ta/train/0.parquet",
    "en": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.simple/train/0.parquet",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for lang, url in SHARDS.items():
        dest = args.out / f"{lang}.parquet"
        if dest.exists():
            print(f"{dest} exists, skipping")
            continue
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

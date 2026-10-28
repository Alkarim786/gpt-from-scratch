"""Byte-level BPE tokenizer, trained from scratch.

* Text is split into pre-tokens by a regex, then each pre-token's UTF-8 bytes
  are merged bottom-up. Every byte is a base token, so any string (Tamil,
  English, emoji, code) round-trips exactly.
* The pre-tokeniser keeps Unicode combining marks (``\\p{M}``) attached to
  their letters. Tamil vowel signs and the virama (புள்ளி) are combining
  marks, so GPT-2's ``\\p{L}+`` pattern would split almost every Tamil
  syllable into separate pre-tokens and BPE could never learn whole words.
* Training uses incremental pair counts plus a lazy max-heap: each merge only
  revisits the words that contain the merged pair, instead of recounting all
  pairs in the corpus.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

# Contractions, letter+mark runs (optionally with one leading space), numbers
# in groups of up to 3, punctuation runs, and whitespace.
PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[\p{L}\p{M}]+| ?\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+""")

EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = merges or []
        self._rebuild()

    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.eot_id = len(self.vocab)  # special token comes after all merges
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1  # + <|endoftext|>

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, verbose: bool = False) -> BPETokenizer:
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 (256 bytes + EOT)")
        word_freq: Counter[str] = Counter()
        for t in texts:
            word_freq.update(PATTERN.findall(t))
        words: list[list[int]] = [list(w.encode("utf-8")) for w in word_freq]
        freqs: list[int] = list(word_freq.values())

        pair_count: defaultdict[tuple[int, int], int] = defaultdict(int)
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs, strict=True)):
            for p in zip(w, w[1:], strict=False):
                pair_count[p] += f
                where[p].add(wi)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        n_merges = vocab_size - 257
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg or -neg <= 0:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)
            touched: dict[tuple[int, int], None] = {}
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:], strict=False):
                    pair_count[p] -= f
                    where[p].discard(wi)
                    touched[p] = None
                words[wi] = w = _merge(w, pair, new_id)
                for p in zip(w, w[1:], strict=False):
                    pair_count[p] += f
                    where[p].add(wi)
                    touched[p] = None
            for p in touched:
                if pair_count[p] > 0:
                    heapq.heappush(heap, (-pair_count[p], p))
                else:
                    pair_count.pop(p, None)
            if verbose and len(merges) % 500 == 0:
                print(f"  merges: {len(merges)}/{n_merges}")
        return cls(merges)

    # ------------------------------------------------------------ encode/decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best = min(zip(ids, ids[1:], strict=False), key=lambda p: self.ranks.get(p, 1 << 30))
            rank = self.ranks.get(best)
            if rank is None:
                break
            ids = _merge(ids, best, 256 + rank)
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        parts = text.split(EOT) if allow_special else [text]
        for i, part in enumerate(parts):
            if i:
                out.append(self.eot_id)
            for chunk in PATTERN.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            if i == self.eot_id:
                buf += EOT.encode()
            else:
                buf += self.vocab[i]
        return buf.decode("utf-8", errors="replace")

    def token_bytes(self, i: int) -> bytes:
        return EOT.encode() if i == self.eot_id else self.vocab[i]

    # ---------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "byte-bpe", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

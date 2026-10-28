import math

import pytest
import torch

from tinygpt.bpe import EOT, PATTERN, BPETokenizer
from tinygpt.data import TokenStream, clean
from tinygpt.generate import SamplingConfig, apply_repetition_penalty, generate, top_k_filter, top_p_filter
from tinygpt.model import GPT, GPTConfig, KVCache
from tinygpt.train import lr_at

CORPUS = [
    "The quick brown fox jumps over the lazy dog. The dog sleeps.",
    "தமிழ் ஒரு செம்மொழி ஆகும். தமிழ்நாடு இந்தியாவின் ஒரு மாநிலம்.",
    "Numbers like 12345 and code like x = y + 1; should round-trip.",
    "சென்னை தமிழ்நாட்டின் தலைநகரம். The capital is Chennai.",
] * 20


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400)


# ----------------------------------------------------------------- tokenizer
def test_pretokenizer_keeps_tamil_marks_attached():
    # "தமிழ்" = த + ம + ி (vowel sign, \p{M}) + ழ + ் (virama, \p{M})
    assert PATTERN.findall("தமிழ் மொழி") == ["தமிழ்", " மொழி"]


def test_roundtrip_any_unicode(tok):
    for s in ["hello world", "தமிழ்நாடு அரசு", "emoji 😀 and ünïcödé", "  spaces\n\nnewlines\t", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_training_compresses_and_is_deterministic(tok):
    text = CORPUS[1]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2
    again = BPETokenizer.train(CORPUS, vocab_size=400)
    assert again.merges == tok.merges
    assert tok.vocab_size == 400


def test_special_token_and_persistence(tok, tmp_path):
    ids = tok.encode(f"a{EOT}b")
    assert tok.eot_id in ids and tok.decode(ids) == f"a{EOT}b"
    assert tok.eot_id not in tok.encode(f"a{EOT}b", allow_special=False)
    tok.save(tmp_path / "t.json")
    loaded = BPETokenizer.load(tmp_path / "t.json")
    assert loaded.encode(CORPUS[3]) == tok.encode(CORPUS[3])


def test_clean_normalises_unicode_and_whitespace():
    decomposed = "é"  # e + combining acute
    assert clean(f"  {decomposed}   x\n\n\n\ny ") == "é x\n\ny"


# --------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=97, block_size=32, n_layer=2, n_head=2, n_embd=32)).eval()


def test_forward_shapes_and_initial_loss(model):
    x = torch.randint(0, 97, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 97)
    assert abs(loss.item() - math.log(97)) < 0.5  # ~uniform at init
    logits, loss = model(x)
    assert logits.shape == (3, 1, 97) and loss is None


def test_causality_future_tokens_dont_affect_past(model):
    x = torch.randint(0, 97, (1, 20))
    y = x.clone()
    y[0, 15:] = (y[0, 15:] + 1) % 97
    lx, _ = model(x, x)
    ly, _ = model(y, y)
    assert torch.allclose(lx[0, :15], ly[0, :15], atol=1e-5)
    assert not torch.allclose(lx[0, 15:], ly[0, 15:], atol=1e-5)


def test_kv_cache_matches_full_forward(model):
    x = torch.randint(0, 97, (1, 24))
    full, _ = model(x, x)  # logits for every position
    cache = KVCache(model.cfg.n_layer)
    model(x[:, :10], cache=cache)  # prefill 10 tokens
    for t in range(10, 24):
        step, _ = model(x[:, t : t + 1], cache=cache)
        assert torch.allclose(step[0, -1], full[0, t], atol=1e-4), f"mismatch at position {t}"
    assert cache.length == 24


def test_cached_and_uncached_greedy_generation_agree(model):
    cfg = SamplingConfig(max_new_tokens=20, temperature=0, top_k=None, top_p=None, repetition_penalty=1.0)
    prompt = [5, 6, 7]
    assert generate(model, prompt, cfg, use_cache=True) == generate(model, prompt, cfg, use_cache=False)


def test_generation_past_block_size_slides_window(model):
    cfg = SamplingConfig(max_new_tokens=80, temperature=1.0, seed=1)
    out = generate(model, [1, 2, 3], cfg)
    assert len(out) == 83 and all(0 <= t < 97 for t in out)


def test_block_size_is_enforced(model):
    with pytest.raises(ValueError):
        model(torch.zeros((1, 33), dtype=torch.long))


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))
    opt = m.configure_optimizer(3e-3, 0.0)
    x = torch.randint(0, 50, (4, 16))
    first = None
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_weight_decay_only_on_matrices():
    m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, n_embd=16))
    decay, no_decay = m.configure_optimizer(1e-3, 0.1).param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert m.lm_head.weight is m.wte.weight  # tied


# ------------------------------------------------------------------ sampling
def test_top_k_and_top_p():
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    assert torch.isfinite(top_k_filter(logits, 2)).sum() == 2
    probs = torch.softmax(logits, -1)  # [0.636, 0.234, 0.086, 0.032, 0.012]
    # 0.636 + 0.234 = 0.870 >= 0.85, so the nucleus is the first two tokens
    assert torch.isfinite(top_p_filter(logits, 0.85)).tolist() == [True, True, False, False, False]
    assert torch.isfinite(top_p_filter(logits, 0.9)).tolist() == [True, True, True, False, False]
    assert torch.isfinite(top_p_filter(logits, 0.5)).sum() == 1
    assert probs.sum().item() == pytest.approx(1.0)


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([2.0, -1.0, 0.5])
    out = apply_repetition_penalty(logits, torch.tensor([0, 1]), 2.0)
    assert out.tolist() == [1.0, -2.0, 0.5]


def test_seeded_sampling_is_reproducible(model):
    cfg = SamplingConfig(max_new_tokens=15, temperature=0.9, seed=42)
    assert generate(model, [3], cfg) == generate(model, [3], cfg)


# ---------------------------------------------------------------- data/train
def test_token_stream_shifts_targets(tmp_path):
    import numpy as np

    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "t.bin")
    x, y = TokenStream(tmp_path / "t.bin", block_size=8, seed=0).batch(4)
    assert x.shape == (4, 8) and torch.equal(y, x + 1)


def test_lr_schedule():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(55, 1.0, 10, 100) == pytest.approx(0.55)

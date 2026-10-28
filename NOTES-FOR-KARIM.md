# Notes for Karim: tinygpt (GPT from scratch)

## Resume bullets (real measured numbers)
- Implemented a **GPT-style transformer, byte-level BPE tokenizer, KV cache and sampling from scratch** in PyTorch. Trained a 5.3M-parameter model on Tamil + English Wikipedia (4M tokens) on CPU, reducing validation perplexity **from 8,474 to 154**.
- Designed a **Tamil-aware BPE pre-tokenizer** (keeps Unicode combining marks with their letters), reaching **3.28 chars/token on Tamil vs 3.25 on English** with a shared 8K vocabulary. Evaluated in tokenizer-independent bits per byte (**Tamil 0.88, English 2.18**).
- Built a **KV cache verified equivalent to full recomputation**, giving a **2.8× generation speed-up** (19 → 54 tokens/s), with 18 tests covering causality, cache equivalence, sampling maths and reproducibility.

## 5 likely interview questions
1. **"Walk me through self-attention."** Q, K, V projections; scores = QKᵀ/√d (scaling keeps softmax from saturating); causal mask sets future positions to −∞; softmax; weighted sum of V; concatenate heads; output projection. Point to `CausalSelfAttention.forward`.
2. **"What does the KV cache do and why is it faster?"** At step t the keys/values of tokens 0..t−1 don't change, so store them and compute only the new token's Q, K, V. Per-step cost drops from processing T tokens to processing 1 (attention is still O(T) per step). The 2.8× measured here grows with context length. Memory cost = 2 × layers × heads × head_dim × T per sequence.
3. **"How does BPE work, and why byte-level?"** Start from 256 byte tokens, repeatedly merge the most frequent adjacent pair. Byte-level means no unknown tokens ever, since any UTF-8 string round-trips. Then explain the Tamil combining-mark issue: it's the strongest original point in this project.
4. **"Top-k vs top-p vs temperature?"**
   - Temperature divides logits: <1 sharpens, >1 flattens.
   - Top-k keeps the k most likely tokens.
   - Top-p keeps the smallest set whose probability reaches p, so it adapts to confidence.
   - Repetition penalty down-weights tokens already generated.
5. **"Why bits per byte instead of perplexity?"** Perplexity is per token, so a tokenizer with longer tokens gets a worse perplexity for the same model quality. BPB normalises by raw bytes: total loss in bits / bytes. That's why Tamil (PPL 186) actually scores *better* than English (PPL 134) per byte.

## Understand this before interviews
- **The causal mask with a cache** (`model.py`): `q_pos = past + arange(T)`, `k_pos = arange(S)`, mask where `k_pos > q_pos`. Without a cache past = 0 and this is the usual triangle; with a cache it lets new queries see all past keys.
- **Weight tying:** `lm_head.weight = wte.weight` saves 2M parameters here (vocab 8,192 × 256) and usually helps small models.
- **Weight decay groups:** decay only ≥2-D tensors (matrices); not biases or LayerNorm gains.
- **The `.view` vs `.reshape` bug:** `x[:, 1:]` is non-contiguous, so `.view(-1)` throws. A good small debugging story.
- **Honest limitations:** English samples are mostly noise at this scale; the Tamil sample is fluent but copies the census-village template. Be ready to say how you'd scale (bigger model, more data, GPU, RoPE, FlashAttention).

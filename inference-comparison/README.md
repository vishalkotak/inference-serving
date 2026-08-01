# inference-comparison

From-scratch replicas of the mechanisms that make **vLLM** and **SGLang** fast, built
component by component against one shared toy model.

The point is not model quality — weights are random and the tokenizer is raw bytes.
The point is the *serving-engine machinery*: how the KV cache is laid out, how blocks
are shared, how a grammar constrains sampling. Every component is validated on its own
terms rather than by eyeballing generated text.

## Why a shared model

`shared/mini_llm.py` is a 2-layer decoder-only Transformer whose attention is delegated
to a pluggable `AttentionBackend`. The same `TinyLLM` is meant to run under vLLM's
PagedAttention backend and SGLang's RadixAttention backend — exactly like a real engine
swapping its KV-cache backend under a fixed model. `DenseBackend` keeps KV in plain
per-`(seq, layer)` lists and exists purely as ground truth: any clever backend must
reproduce its numbers exactly.

## Status

| Component | What it does | State |
| --- | --- | --- |
| `shared/mini_llm.py` | TinyLLM, byte tokenizer, `AttentionBackend` protocol, `DenseBackend` reference | done |
| `shared/regex_fsm.py` | regex → NFA (Thompson) → DFA over the byte alphabet; exposes `allowed_bytes` / `is_accepting` / `step` | done |
| `vllm_replica/c1_paged_kv_cache.py` | `BlockAllocator` + `PagedKVCache`: block pool, per-sequence block tables, scatter-write / gather-read, refcounted free | done |
| `vllm_replica/c2_prefix_cache.py` | `PrefixCache`: cumulative-prefix → physical block map, exact-prefix reuse, pinning | done |
| PagedAttention backend | wire `PagedKVCache` into the `AttentionBackend` protocol and assert equality with `DenseBackend` | **not started** |
| SGLang replica | RadixAttention (radix-tree prefix sharing) | not started |
| Structured decoding | drive `RegexFSM` from the sampler to mask illegal tokens | not started |

`PagedKVCache` currently stands alone — it stores and returns KV correctly, but nothing
runs a forward pass through it yet. Closing that gap is the next step, and it is what
makes the `DenseBackend` equality check meaningful.

## Design notes worth keeping

**One block holds every layer.** The pool is
`blocks[num_blocks, n_layer, 2, block_size, n_head, d_head]`. Keeping all layers inside
one physical block means a sequence needs exactly *one* block table for the whole model
rather than one per layer — the same choice real vLLM makes, and it keeps the
bookkeeping trivial.

**Address translation is the whole trick.** For an absolute token position `p`:
`logical_block = p // block_size`, `slot = p % block_size`,
`physical = block_table[logical_block]`. Two lines of arithmetic buy an arbitrary,
non-contiguous physical layout, so a sequence grows one block at a time and freed
blocks are instantly reusable.

**Prefix keys are cumulative, not per-block.** `PrefixCache` keys each block on *all*
tokens up to and including it, so a hit implies every earlier token matched too. That
makes collisions impossible and enforces the real constraint of automatic prefix
caching: reuse must be contiguous from position 0. A prompt that shares text starting
at position 3 reuses nothing.

**Sharing needs refcounts.** Once two sequences point at the same physical block,
freeing one must not hand that block back to the pool. `pin()` adds a reference so
prefix blocks outlive the request that produced them.

## Layout

```
inference_comparison/
  shared/
    mini_llm.py            TinyLLM + AttentionBackend protocol + DenseBackend
    regex_fsm.py           regex -> DFA over bytes
  vllm_replica/
    c1_paged_kv_cache.py   BlockAllocator, PagedKVCache
    c2_prefix_cache.py     PrefixCache
  quick_check.py           smoke test: model forward
  check_fsm.py             smoke test: regex -> DFA
  check_init.py            smoke test: package imports
  inference_comparison.ipynb   working scratchpad (components are derived here first)
tests/                     pytest suite
```

## Running it

```bash
poetry install --with dev
poetry run pytest
```

The three `check_*.py` scripts are older standalone smoke tests and still run directly:

```bash
cd inference_comparison && poetry run python quick_check.py
```

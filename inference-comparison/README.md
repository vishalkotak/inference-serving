# inference-comparison

From-scratch replicas of the mechanisms that make vLLM and SGLang fast, built one
component at a time against a shared toy model.

Weights are random and the tokenizer is raw bytes, so the generated text is
meaningless. What gets validated is the machinery: KV cache layout, block sharing,
grammar-constrained sampling.

## Status

| Component | State |
| --- | --- |
| `shared/mini_llm.py` — TinyLLM, byte tokenizer, `AttentionBackend` protocol, `DenseBackend` reference | done |
| `shared/regex_fsm.py` — regex to DFA over the byte alphabet | done |
| `vllm_replica/c1_paged_kv_cache.py` — block pool, block tables, refcounted free | done |
| `vllm_replica/c2_prefix_cache.py` — exact-prefix block reuse | done |
| PagedAttention backend — run TinyLLM on `PagedKVCache`, check it matches `DenseBackend` | next |
| SGLang replica — RadixAttention | todo |
| Structured decoding — drive `RegexFSM` from the sampler | todo |

`PagedKVCache` stores and returns KV correctly, but nothing runs a forward pass through
it yet. That is the next step, and it is what makes the `DenseBackend` comparison mean
anything.

## Notes

All layers share one block, so a sequence needs one block table for the whole model
instead of one per layer. Same choice vLLM makes.

Position `p` lives at block `block_table[p // block_size]`, slot `p % block_size`. That
indirection is the entire trick — blocks can sit anywhere in the pool.

`PrefixCache` keys each block on every token up to and including it, so a hit means the
whole prefix matched. Reuse has to start at position 0; a prompt sharing text from
position 3 reuses nothing.

Shared blocks are refcounted. `pin()` keeps prefix blocks alive after the request that
filled them finishes.

## Running

    poetry install --with dev
    poetry run pytest

`inference_comparison/inference_comparison.ipynb` is the scratchpad where components
get worked out before they land in files.

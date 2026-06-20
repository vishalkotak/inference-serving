# Day 01 — Why `vllm serve` failed before serving anything (KV-cache preflight)

When I tried to serve `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` with no flags, vLLM
**failed a preflight feasibility check at startup — not a runtime allocation.** The model loaded
fine; the engine then refused to start because the KV-cache pool couldn't guarantee even one
full-length request.

## The error

Key line:

```
ValueError: To serve at least one request with the model's max seq len (131072),
16.0 GiB KV cache is needed, which is larger than the available KV cache memory (8.99 GiB).
Based on the available memory, the estimated maximum model length is 73680.
```

The `vllm-metal` backend logged exactly how it budgeted memory just before failing:

```
Paged attention memory breakdown:
  metal_limit=17.18GB, fraction=0.9, usable_metal=15.46GB,
  model_memory=4.52GB, overhead=1.29GB, kv_budget=9.66GB,
  per_block_bytes=2097152, num_blocks=4605, max_tokens_cached=73680
KV cache: 9657.4 MB (32 layers, 4605 blocks, 16 tokens/block)
```

<details>
<summary>Full traceback</summary>

```
(EngineCore) ... _initialize_kv_caches(vllm_config)
  -> get_kv_cache_configs(...)
  -> _check_enough_kv_cache_memory(...)
  -> ValueError: 16.0 GiB needed > 8.99 GiB available  (est. max len 73680)
(APIServer) RuntimeError: Engine core initialization failed.
            Failed core proc(s): {'EngineCore': 1}
```
</details>

## Why it happened

1. **`max-model-len` not specified → vLLM adopts the model's full context window.** It reads
   `max_position_embeddings` from the checkpoint config, and `Llama-3.1-8B-Instruct` is a
   **128K (131072)** context model. So that became `max_model_len` by default.
2. **vLLM guarantees it can serve at least one request at that length before it starts.** On boot
   it asks: *"can my KV-cache pool hold a single sequence of `max_model_len` tokens?"* If not, it
   aborts rather than start a server that would crash on the first long request.

## How many bytes does one token add to the cache?

| Factor      | Value | Why it's there                                                                                            |
| ----------- | ----- | --------------------------------------------------------------------------------------------------------- |
| K and V     | 2     | You cache two vectors per token — one Key, one Value                                                       |
| layers      | 32    | Each of Llama-8B's 32 transformer layers has its own independent attention, so its own K/V                |
| kv_heads    | 8     | Attention is split into heads. Llama-3.1 uses GQA, which keeps only 8 KV heads (not 32) — memory saving    |
| head_dim    | 128   | Each head's vector is 128 numbers long                                                                     |
| dtype_bytes | 2     | Each number is stored in fp16 = 2 bytes                                                                    |

- Cost of one token = **2 × 32 × 8 × 128 × 2 = 131072 bytes = 128 KiB**
- Full context = **131072 tokens** (the 128K window)
- Total needed = 131072 tokens × 128 KiB = **16 GiB**  ← matches the error exactly

> Note: the "131072 bytes/token" and "131072 tokens" being equal is a coincidence — one is
> `2×32×8×128×2`, the other is the 128K window. Don't read meaning into the match.

## But we're using 4-bit quantization?

- 4-bit shrank the **weights** (`model_memory=4.52GB`), which is why the model loaded fine.
- The **KV cache is still fp16** (`dtype_bytes = 2`). Quantizing the weights does nothing to it.
- So on a 128K model the **KV cache, not the weights, is the memory hog**: 16 GiB for one
  full-length sequence vs. ~4.5 GB for the entire model.

## Where the 8.99 GiB "available" came from

Follow the backend's breakdown:

```
metal_limit 17.18GB × fraction 0.9 = usable_metal 15.46GB
15.46GB − model_memory 4.52GB − overhead 1.29GB = kv_budget 9.66GB
9.66GB / per_block_bytes 2,097,152 (2 MB) = 4605 blocks
4605 blocks × 16 tokens/block = 73,680 tokens cacheable
```

- `kv_budget = 9.66 GB` (decimal) = **8.99 GiB** (binary) — same number, just GB vs GiB. That's the
  "available" figure in the error.
- `max_tokens_cached = 73680` is the *total* token budget across all sequences, and exactly the
  "estimated maximum model length" the error suggests.

## The fix

Cap the context so the guarantee fits the pool:

```bash
vllm serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --max-model-len 8192 --gpu-memory-utilization 0.45
```

- `--max-model-len` lowers the promise (one 8K sequence needs ~1 GiB, not 16).
- `--gpu-memory-utilization` controls how much of unified memory the pool may claim — kept low on a
  shared 24 GB Mac so the OS isn't starved into swap (see the lag incident in the Day 1 work).

See also: [`../bench/d1_loadtest.py`](../bench/d1_loadtest.py),
[`../results/d1-load-test.md`](../results/d1-load-test.md).

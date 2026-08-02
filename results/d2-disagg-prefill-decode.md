# Day 02: Disaggregated prefill/decode on 2× T4, partial result

**Goal:** reproduce the llm-d disaggregated architecture's core mechanism (prefill builds the KV
cache on one GPU, decode consumes it on another) on free hardware, without Kubernetes.

**Outcome: the pipeline stands up end to end, but the KV handoff does not complete.** The final run
returns HTTP 200 with a coherent completion while both engines log `LMCache hit tokens: 0`. That is
the decoder recomputing the prompt locally, not a transfer. Recording it as a negative result
because the failure mode is the interesting part: **every wrong configuration on this path returns
a correct-looking answer.**

## Setup

| | |
|---|---|
| Hardware | Kaggle notebook, **2× Tesla T4** (16 GB each, Turing → fp16, no bf16, no FA2) |
| Engine | vLLM `0.26.0` |
| Transfer | LMCache `0.5.2-gcd2c0d6a`, `LMCacheConnectorV1`, `transfer_channel: nixl` |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, `--dtype half --enforce-eager --max-model-len 2048` |
| Topology | prefiller `:8100` (GPU 0, sender) → decoder `:8200` (GPU 1, receiver) → proxy `:9000` |
| Proxy | vLLM's own `disagg_proxy_server.py` (a hand-rolled one is what caused failure #2 below) |

Launch order matters: **decoder first** (it binds the PD ports), then prefiller (it dials), then
proxy. Both at `--gpu-memory-utilization 0.6`, since 0.8 leaves no room for LMCache's 1 GiB PD
buffer.

## What worked

- Two independent vLLM engines, one per T4, `GPU KV cache size: 317,952 tokens` each.
- LMCache PD config **accepted** after the schema migration: no `Unknown configuration key`
  warnings, dump shows `enable_pd: True`, `pd_role`, `transfer_channel: nixl`, all three ports.
- NIXL initialises on both sides: `Backend UCX was instantiated`, `Initialized NIXL agent`, and the
  receiver reaches `Starting async initialization loop (nixl_channel.py:344)`.
- A 2007-token request routes through the official proxy and reaches both engines.

## What didn't

```
===== PREFILLER (sender) =====
LMCache INFO: Reqid: cmpl-89a7a256af64ca1f-0, Total tokens 2007,
              Inference Engine computed tokens: 0, LMCache hit tokens: 0, need to load: 0
LMCache WARNING: LMCache is unhealthy, skipping store operation

===== DECODER (receiver) =====
NIXL INFO  _api.py:268 Initialized NIXL agent: d7100b13-...
LMCache INFO: Starting async initialization loop (nixl_channel.py:344)
LMCache INFO: Reqid: cmpl-84c603df084e4c86-0, Total tokens 2007,
              Inference Engine computed tokens: 0, LMCache hit tokens: 0, need to load: 0
```

Both agents init, the async channel starts, and the sender→receiver handshake never completes, so
nothing is staged, and `hit tokens` stays 0 on both ends.

## Findings

- **The success signal is `hit tokens`, not the HTTP response.** Four distinct misconfigurations on
  this path (removed connector, port collision, ignored config schema, sub-chunk prompt) all
  produced fluent, plausible completions. Local recompute is the *designed* fallback, which makes
  the output text useless as a verification signal. The win condition is the decoder logging
  `LMCache hit tokens: ~2000, need to load: ~2000` with no `unhealthy`.
- **`chunk_size: 256` sets a floor on what can transfer at all.** The first tests used
  `"The capital of France is"`, which is 5 tokens, below one chunk, so LMCache stages nothing and
  reports `hit tokens: 0` even on a fully working setup. Indistinguishable from real failure unless
  you know the threshold. vLLM's own disagg benchmark uses `--random-input-len 7500` for exactly
  this reason.
- **`kv_transfer_params: None` is the tell for a broken hand-rolled proxy.** A two-stage proxy
  (`max_tokens=1` to prefill, then forward to decode) returned a correct answer with
  `kv_transfer_params: None`. Decode ignored the handoff entirely. vLLM ships the correct proxy;
  the coordination it does is not obvious enough to reimplement casually.
- **PD mode is designed for two machines.** `pd_peer_host` expects a different box. On one host,
  sender and receiver share `localhost` and the same three ports, and that's where the async init
  loop stalls. Related and unresolved: the receiver's log contains
  `assert config.pd_peer_host is not None`, even though omitting that key is precisely what makes it
  the listener. See the [notes](../notes/d2-lmcache-pd-schema-migration.md).
- **`kill -9` on a vLLM process does not free its GPU memory.** Killed workers strand CUDA
  allocations; the next launch OOMs at `Free memory on device cuda:0 (3.7/14.56 GiB)`. Every
  relaunch needs a full sweep of `nvidia-smi --query-compute-apps=pid`, then a check that free
  memory is back to ~14.9 GiB.

## Next

Two ways to finish this, in increasing order of fidelity:

1. **Swap the PD transport to `local_cpu`.** KV still crosses prefill → decode, via host memory
   instead of NIXL. More reliable on a single node and would light up `hit tokens`, proving the
   handoff, just not "over NIXL". Cheapest way to close the loop on the mechanism.
2. **Deploy across two nodes** (two Modal GPU functions or RunPod boxes) so `pd_peer_host` is a real
   remote, the configuration LMCache actually tests. Everything established here carries over
   unchanged: the migrated schema, `save_unfull_chunk`, listener-before-dialer ordering, matched
   `PYTHONHASHSEED`, the >1-chunk prompt requirement.

Full run with preserved outputs and the failed attempts:
[`../disagg-prefill-decode/disaggregated-prefill-decode-vllm-lmcache.ipynb`](../disagg-prefill-decode/disaggregated-prefill-decode-vllm-lmcache.ipynb).

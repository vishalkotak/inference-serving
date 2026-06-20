# Day 01 — Concurrency load test: throughput vs. latency

**Goal:** serve Llama-3.1-8B-Instruct and prove the throughput/latency tradeoff under
concurrency — the baseline for the continuous-batching work later in the sprint.

## Setup

| | |
|---|---|
| Hardware | MacBook Pro, Apple M4 Pro, 24 GB unified memory |
| Engine | `vllm-metal` (vLLM + MLX/Metal backend) |
| Model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (4-bit) |
| Server flags | `--max-model-len 8192 --gpu-memory-utilization 0.45` |
| Harness | [`bench/d1_loadtest.py`](../bench/d1_loadtest.py) — stdlib threads, non-streaming, `max_tokens=128` |

```bash
python3 bench/d1_loadtest.py --n 20 --concurrency 1
python3 bench/d1_loadtest.py --n 20 --concurrency 4
python3 bench/d1_loadtest.py --n 20 --concurrency 8
```

## Raw output

```
$ python3 d1_loadtest.py --n 20 --concurrency 1
Done in 47.56s
System throughput:   33.6 tok/s | Per-request:    33.6 tok/s | Mean latency:  2.38s | P99:  3.57s

$ python3 d1_loadtest.py --n 20 --concurrency 4
Done in 23.04s
System throughput:   69.0 tok/s | Per-request:    17.5 tok/s | Mean latency:  4.56s | P99:  6.16s

$ python3 d1_loadtest.py --n 20 --concurrency 8
Done in 22.43s
System throughput:   74.6 tok/s | Per-request:    10.7 tok/s | Mean latency:  8.08s | P99: 11.55s
```

## Results

| Concurrency | Wall | System tput | Per-req tput | Mean latency | P99 |
|---|---|---|---|---|---|
| 1 | 47.6s | 33.6 tok/s | 33.6 tok/s | 2.38s | 3.57s |
| 4 | 23.0s | **69.0 tok/s** | 17.5 tok/s | 4.56s | 6.16s |
| 8 | 22.4s | **74.6 tok/s** | 10.7 tok/s | 8.08s | 11.55s |

## Findings

- **System throughput roughly doubled from concurrency 1 → 4** (33.6 → 69.0 tok/s). The GPU
  advances multiple sequences per decode step, so total tokens/sec rises even though no single
  request gets faster — continuous batching in action.
- **Per-request throughput collapsed** (33.6 → 17.5 → 10.7 tok/s). One sequence can't decode any
  faster, and now it shares the GPU and waits its turn in each batched step.
- **Saturation knee is between 4 and 8.** From 4 → 8, system throughput barely moved (+8%,
  69.0 → 74.6) while mean latency nearly doubled (4.56 → 8.08s) and P99 went 6.2 → 11.6s. Past the
  knee, extra concurrency buys almost no throughput and just adds queueing delay — i.e. this is
  roughly where a production max-concurrency limit should sit.
- **Sanity check:** per-request × concurrency ≈ system throughput. At c=4: 17.5 × 4 = 70 ≈ 69 ✓.
  At c=8: 10.7 × 8 = 85.6 vs. 74.6 measured — the gap is because with only 20 requests the run
  can't keep all 8 slots full through the tail, so *effective* concurrency was below 8.

## Caveats / next

- **This ~70 tok/s ceiling is a Metal/4-bit limit, not a vLLM limit.** On an L4 (bf16, dedicated
  VRAM, higher memory bandwidth) the knee sits much higher and batching keeps scaling past c=8.
  Running this same harness on an L4 is the cleanest way to show the hardware dependence.
- **P99 at `--n 20` is noisy** (essentially the slowest of 20 requests). Re-run at `--n 200+` for a
  trustworthy tail percentile.

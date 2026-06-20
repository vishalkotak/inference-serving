# inference-serving

An 8-week, artifact-driven sprint to go from "can serve a model" to deep,
interview-ready LLM inference-serving expertise. One growing repo, shipped daily.

The plan is tracked in [`tracker.html`](tracker.html) (open it in a browser —
progress persists locally via `localStorage`).

## Sprint arc

| Weeks | Focus |
|------|-------|
| 1 | Core inference mechanics — serve, benchmark, prove bottlenecks with the roofline model |
| 2 | Production serving + first modality — prefix caching, quantization + evals, K8s, observability, embeddings |
| 3 | Serving at scale — multi-replica, load/prefix-aware routing, autoscaling, zero-downtime ops |
| 4 | KV cache mastery — blocks, cross-request reuse, tiering, quantization |
| 5 | Disaggregation & advanced scheduling — split prefill/decode, SLO-aware scheduling |
| 6 | GPU kernels — profiling, FlashAttention, Triton, quant kernels |
| 7 | Frontier-scale distributed serving — tensor / pipeline / expert parallelism, comms |
| 8 | Capstone — one platform, the deep-dive writeup, an OSS PR |

## Repository layout

```
bench/          load/concurrency/latency benchmark scripts
benchmarks/     consolidated reproducible benchmark suite (week 8)
tools/          kv_cache_calc.py and other calculators
evals/          reusable quality/eval harness (built wk2, reused throughout)
results/        dayNN-*.md writeups + chart PNGs (the portfolio surface)
notes/          conceptual writeups (forward pass, vLLM internals, KV blocks, kernels)
gateway/        FastAPI gateway in front of vLLM
router/         multi-replica routing policies
embeddings/     embedding-model serving (non-LLM modality)
observability/  Grafana dashboards + SLO docs
disagg/         disaggregated prefill/decode setup
kernels/        kernel profiling + Triton experiments
distributed/    tensor / pipeline / expert parallelism studies
platform/       week-8 integrated platform
k8s/            deployment manifests (scale/ autoscale/ rollout/)
```

Top-level deep-dives are added as the sprint progresses:
`ARCHITECTURE.md`, `SCALING.md`, `SCHEDULING.md`, `KV-CACHE.md`, `KERNELS.md`,
`DISTRIBUTED.md`, `RESULTS.md`, `BLOG.md`, `TALK.md`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in HF_TOKEN, etc.
```

## Conventions

- Commit daily; one day's work per commit, e.g. `day04: roofline proof — decode is memory-bound`.
- Charts and writeups live under `results/`; raw model weights, datasets, and
  profiler traces are **not** committed (see `.gitignore`).

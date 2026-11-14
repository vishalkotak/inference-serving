# inference-serving

How LLM serving engines get their speed, rebuilt and measured.

| Path | |
|---|---|
| [inference-comparison](inference-comparison/) | vLLM's paged KV cache and prefix cache from scratch, tested against a dense reference. |
| [bench](bench/) | Concurrency load tester. |
| [disagg-prefill-decode](disagg-prefill-decode/) | vLLM + LMCache prefill/decode split across two GPUs. |

Results
- [Batching knee](results/d1-load-test.md): Llama 3.1 8B on M4 Pro. Throughput flat past 4 concurrent requests while latency doubles.
- [Disaggregated prefill/decode](results/d2-disagg-prefill-decode.md): pipeline runs, KV handoff doesn't. Why HTTP 200 proves nothing here.

#!/usr/bin/env python3
"""Day 1 load test — fire concurrent requests at the vLLM server, measure throughput + latency.

  Prereq: server up in another terminal:
    vllm serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
      --max-model-len 8192 --gpu-memory-utilization 0.45

  Run:
    python3 d1_loadtest.py --n 20 --concurrency 1     # baseline, one at a time
    python3 d1_loadtest.py --n 20 --concurrency 4     # then push concurrency up
    python3 d1_loadtest.py --n 20 --concurrency 8
"""

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

def fire_one_request(prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        }).encode()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    start_time = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read())
    latency = time.perf_counter() - start_time
    return {"latency": latency, "output_tokens": data["usage"]["completion_tokens"]}

def run_load(n: int, concurrency: int, max_tokens: int):
    prompts = [f"In two sentences, explain why the number {i} is interesting." for i in range(n)]
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda p: fire_one_request(p, max_tokens), prompts))
    latency = time.perf_counter() - start_time
    return results, latency

def summarize(results: list[dict], wall_clock: float) -> None:
    output_tokens_sum = 0
    latency_sum = 0.0
    per_request_tput_sum = 0.0
    request_latencies = []
    for r in results:                       
        latency = r["latency"]
        tokens = r["output_tokens"]
        output_tokens_sum += tokens
        latency_sum += latency
        per_request_tput_sum += tokens / latency
        request_latencies.append(latency)

    n = len(results)
    request_latencies.sort()            
    index = math.ceil(0.99 * n) - 1
    print(f"System throughput: {output_tokens_sum / wall_clock:6.1f} tok/s | "
          f"Per-request:  {per_request_tput_sum / n:6.1f} tok/s | "
          f"Mean latency: {latency_sum / n:5.2f}s | "
          f"P99: {request_latencies[index]:5.2f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="total requests to send")
    ap.add_argument("--concurrency", type=int, default=4, help="max requests in flight at once")
    ap.add_argument("--max-tokens", type=int, default=128, help="output tokens per request")
    args = ap.parse_args()
    print(f"Firing {args.n} requests at concurrency {args.concurrency} "
          f"(max_tokens={args.max_tokens})...")

    results, wall = run_load(args.n, args.concurrency, args.max_tokens)
    print(f"Done in {wall:.2f}s\n")
    summarize(results, wall)

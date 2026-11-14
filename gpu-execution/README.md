# gpu-execution

Notebooks working through how GPU inference actually executes, run on
remote NVIDIA hardware from a Mac using [Modal](https://modal.com).

Each directory is a self-contained experiment: a question, a prediction,
a measured run, and a conclusion.

## Tracks

### `01-remote-gpu-smoke-test/`

Establishes the baseline: can a Python function defined locally run on a
remote NVIDIA L4, and what is actually where?

Builds a Modal image from `nvidia/cuda:12.8.1-base-ubuntu22.04`, requests
an L4, and parses `nvidia-smi` for name, driver version, total memory, and
compute capability. The verified environment is recorded in
`artifacts/gpu_environment.json`.

The point of the exercise is the split between host and container:

- The GPU belongs to the physical host, not the container.
- The container receives GPU device files and driver libraries.
- The CUDA image supplies user-space CUDA; the kernel driver runs on the host.
- Defining a Modal function allocates nothing. The GPU is requested when
  `.remote()` executes.

### `02-cuda-execution-model/`

Asks what a CPU process is really doing when it submits work to a GPU.

Times a single 4096x4096 `torch.mm` on an L4 three ways, after five warm-up
iterations:

| Measurement | Result |
| --- | --- |
| Host-side launch | 0.086 ms |
| Wall clock around `torch.cuda.synchronize()` | 11.81 ms |
| CUDA events | 12.08 ms |

The first number is roughly 140x smaller because `torch.mm` returns once the
operation has been submitted to the CUDA stream, not once the GPU has finished
it. An unsynchronized Python timer therefore measures dispatch overhead.

The synchronized and event-based numbers agree, which is what makes them
trustworthy: the multiplication takes about 12 ms, and submitting it costs
under 0.1 ms.

The conclusion worth carrying forward is that CUDA execution is asynchronous,
not lazy. The work starts on submission. `torch.cuda.synchronize()` does not
trigger it, it blocks the CPU until already-submitted work completes.

### `03-cuda-streams-and-overlap/`

Takes the previous result further: if the CPU runs ahead of the GPU, can two
GPU operations run at the same time, and does that make anything faster?

**Same-stream ordering.** Two 4096x4096 products submit in 0.196 ms while the
GPU spends 11.67 ms and 11.66 ms executing them. The two intervals sum to the
total to within 0.0 ms, so the stream serialised the work even though the CPU
had already moved on. One stream is an ordered queue.

**Two streams, large operations.** The kernels do overlap, by roughly 10.83 ms,
but operation B stretches from 11.66 ms to 21.84 ms under contention:

| | Single op | Two streams |
| --- | --- | --- |
| Operation A | 11.67 ms | 11.43 ms |
| Operation B | 11.66 ms | 21.84 ms |
| Total | 23.33 ms | 22.44 ms |

Concurrency, but essentially no throughput gain. One 4096x4096 product already
occupies most of the GPU.

**Two streams, small operations.** 100 half-precision 1024x1024 products per
stream, where each kernel is small enough to leave the GPU underused. The first
attempt measured 0.79x, but that was a measurement bug rather than a result:
the CPU queued all of stream A before starting stream B, so the benchmark
captured host-side submission skew. Interleaving the submissions gives 1.01x,
which is a tie.

The track ends on the distinction that matters for serving: multiple streams
create the *possibility* of concurrent execution, but concurrency is not extra
compute capacity. Streams pay off when the operations need complementary
resources, and throughput has to be judged on work completed rather than on
whether two timelines happen to overlap.

There is also a long reference section on `record()`, `wait_event()`, and the
three levels of `synchronize()`, covering which of them block the CPU and which
only express a dependency between GPU streams.

## Running these

```bash
pip install "modal>=1.5,<2"
modal setup
```

Then open either notebook and run the cells. Both request an `L4` and will
allocate a GPU, so runs are billable on your Modal workspace.

Committed notebook outputs are from real runs against an NVIDIA L4.

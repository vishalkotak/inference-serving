# gpu-inference-systems

Notebooks working through how GPU inference actually executes, run on
remote NVIDIA hardware from a Mac using [Modal](https://modal.com).

Each directory is a self-contained experiment: a question, a prediction,
a measured run, and a conclusion.

## Tracks

### `remote-gpu-smoke-test/`

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

### `cuda-execution-model/`

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

## Running these

```bash
pip install "modal>=1.5,<2"
modal setup
```

Then open either notebook and run the cells. Both request an `L4` and will
allocate a GPU, so runs are billable on your Modal workspace.

Committed notebook outputs are from real runs against an NVIDIA L4.

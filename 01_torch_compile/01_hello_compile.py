"""
01 — Hello, torch.compile

Concept:
    torch.compile traces your function on first call, compiles a fused
    Triton kernel, and caches it. Subsequent calls are much faster because
    (a) Python overhead is gone — the whole graph is one CUDA launch,
    (b) Inductor fuses elementwise ops so memory bandwidth is the only cost.

What to notice when you run this:
    - The FIRST compiled call is SLOWER than eager (compile cost).
    - After warmup, compiled is faster, often 1.5–4x for memory-bound ops.
    - The gap widens with more chained pointwise ops (more fusion wins).
"""

import time
import torch


def fused_pointwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 5 pointwise ops. In eager mode each one is a separate CUDA kernel
    # that reads & writes the full tensor to HBM. Inductor fuses all of
    # them into ONE kernel that reads x,y once and writes the result once.
    a = x * y
    b = a + x
    c = torch.sin(b)
    d = c * c
    return d + y


def benchmark(fn, x, y, iters=100):
    # Warmup is essential. The first compiled call pays the compile cost.
    for _ in range(10):
        fn(x, y)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn(x, y)
    torch.cuda.synchronize()  # GPU work is async; sync before stopping the clock.
    return (time.perf_counter() - start) / iters * 1e6  # microseconds per call


def main():
    assert torch.cuda.is_available(), "This lesson needs a CUDA GPU."
    device = "cuda"
    x = torch.randn(1 << 22, device=device)  # 4M floats = 16 MB
    y = torch.randn_like(x)

    eager = fused_pointwise
    compiled = torch.compile(fused_pointwise)

    # Force the first compile so timing below is the steady-state cost.
    print("Compiling...")
    t0 = time.perf_counter()
    compiled(x, y)
    torch.cuda.synchronize()
    print(f"  first call (compile + run): {time.perf_counter() - t0:.3f} s\n")

    eager_us = benchmark(eager, x, y)
    compiled_us = benchmark(compiled, x, y)

    print(f"Eager    : {eager_us:7.2f} us/call")
    print(f"Compiled : {compiled_us:7.2f} us/call")
    print(f"Speedup  : {eager_us / compiled_us:.2f}x")

    # Sanity check that outputs match.
    torch.testing.assert_close(eager(x, y), compiled(x, y))
    print("\nOutputs match.")


if __name__ == "__main__":
    main()

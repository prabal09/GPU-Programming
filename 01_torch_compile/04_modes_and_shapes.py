"""
04 — Compile modes and dynamic shapes

Concept A: mode=
    torch.compile takes a `mode` argument that picks a tradeoff preset.

    "default"          : fast compile, good speedup. Use during development.
    "reduce-overhead"  : uses CUDA Graphs to eliminate per-call launch overhead.
                         Big win for SMALL models / SMALL batches where Python +
                         driver overhead dominates. Costs more memory.
    "max-autotune"     : tries many Triton block sizes / pipelining options and
                         picks the fastest. Slow first compile (seconds–minutes),
                         best steady-state perf. Use for prod / benchmarking.

Concept B: dynamic shapes and recompilation
    Dynamo specializes on input shape by default. If you call the compiled fn
    with a tensor whose shape differs from what it specialized on, it RECOMPILES.

    Recompilation is expensive (hundreds of ms to seconds). If your batch size
    varies, either:
      (a) Pass dynamic=True to torch.compile, OR
      (b) Mark specific dims as dynamic: torch._dynamo.mark_dynamic(x, 0)

    Dynamic compilation produces a kernel that takes shape as a runtime arg —
    slightly slower per call than a fully-specialized one, but you compile once
    and reuse forever.

Watch for the "recompiles" log. In real training loops, mysterious slowdowns
are almost always silent recompiles.
"""

import time
import torch
import torch._logging


def model(x: torch.Tensor) -> torch.Tensor:
    return (x * x + x).sin().cos() * 2.0


def time_one_call(fn, x):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000  # ms


def main():
    assert torch.cuda.is_available()

    # Turn on the recompile log so we can SEE specialization happen.
    torch._logging.set_logs(recompiles=True)

    # ---- Part A: shape specialization (the trap) ----
    print("=== Part A: static (default) compile, varying shapes ===")
    fn_static = torch.compile(model)
    for n in [1024, 1024, 2048, 4096, 2048]:
        x = torch.randn(n, device="cuda")
        ms = time_one_call(fn_static, x)
        print(f"  n={n:5d}  first-time-this-shape? {'maybe' if n in (1024, 2048, 4096) else 'no':5s}  took {ms:7.2f} ms")
    # You'll see big spikes the first time each new shape is seen.
    # The repeat n=1024 and n=2048 should be fast — cached.

    # ---- Part B: dynamic shapes ----
    print("\n=== Part B: dynamic=True, varying shapes ===")
    fn_dyn = torch.compile(model, dynamic=True)
    for n in [1024, 2048, 4096, 8192]:
        x = torch.randn(n, device="cuda")
        ms = time_one_call(fn_dyn, x)
        print(f"  n={n:5d}  took {ms:7.2f} ms   (one compile total, no respecialization)")

    # ---- Part C: modes ----
    print("\n=== Part C: mode comparison (steady-state) ===")
    x = torch.randn(1 << 20, device="cuda")
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        # Fresh compile per mode. Reset the dynamo cache so we're honest.
        torch._dynamo.reset()
        fn = torch.compile(model, mode=mode)
        # Warmup (and compile + autotune cost for max-autotune)
        for _ in range(20):
            fn(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(1000):
            fn(x)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e3  # ms for 1000 calls
        print(f"  mode={mode:18s}  {us:7.2f} ms / 1000 calls")


if __name__ == "__main__":
    main()

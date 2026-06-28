"""
02 — NVTX ranges: annotate the timeline so it makes sense

Concept:
    Without annotations, the Nsight timeline shows raw kernel names like
    `ampere_sgemm_128x64_tn` — accurate but not orienting. NVTX (NVIDIA Tools
    Extension) lets you push named ranges from Python that show up as labeled
    blocks on the timeline.

    Use it to mark phases:
        data load / H2D copy / forward / loss / backward / optimizer.step

    Two APIs in PyTorch:
        torch.cuda.nvtx.range_push("name")   # start
        torch.cuda.nvtx.range_pop()          # end
        # or the context manager:
        with torch.cuda.nvtx.range("name"):  # cleaner; auto-pop
            ...

    Make sure to profile with `-t cuda,nvtx` so nsys captures the ranges.

What you'll see in the GUI:
    A new row labeled "NVTX" with your named blocks. Now when you spot a gap
    on the GPU stream, you can read across to NVTX and see "oh, the gap is
    during the data_load range — that's the bottleneck."

This is THE feature that turns Nsight from "interesting" to "useful" for
debugging a real training loop. Always annotate.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import nvtx


def slow_numpy_batch(batch_size: int, dim: int) -> np.ndarray:
    x = np.random.randn(batch_size, dim).astype(np.float32)
    for _ in range(3):
        x = np.tanh(x) + 0.01 * np.sin(x * 3.14)
    return x


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    batch_size, dim = 256, 4096

    model = nn.Sequential(
        nn.Linear(dim, dim), nn.GELU(),
        nn.Linear(dim, dim), nn.GELU(),
        nn.Linear(dim, dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    target = torch.randn(batch_size, dim, device=device)

    torch.cuda.profiler.start()
    t0 = time.perf_counter()

    for step in range(50):
        # Each `with nvtx.range(...)` produces a labeled block on the NVTX row.
        # Nesting works — you'll see a hierarchy in the GUI.
        with nvtx.range(f"step_{step}"):

            with nvtx.range("data_load"):
                batch_np = slow_numpy_batch(batch_size, dim)

            with nvtx.range("h2d_copy"):
                x = torch.from_numpy(batch_np).to(device)

            with nvtx.range("forward"):
                y = model(x)

            with nvtx.range("loss"):
                loss = loss_fn(y, target)

            with nvtx.range("backward"):
                opt.zero_grad(set_to_none=True)
                loss.backward()

            with nvtx.range("optimizer_step"):
                opt.step()

            # Bad practice — sync inside the loop — but kept here so the timeline
            # clearly shows the sync. Drop this once you're profiling for real.
            with nvtx.range("loss_to_cpu_sync"):
                _ = loss.item()

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"50 steps in {time.perf_counter() - t0:.2f} s")
    print("Profile with:")
    print("  nsys profile -o annotated -t cuda,nvtx -f true python 02_nvtx_annotated.py")
    print("Then open annotated.nsys-rep in the GUI and look at the NVTX row.")


if __name__ == "__main__":
    main()

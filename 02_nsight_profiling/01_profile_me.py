"""
01 — A workload with an intentional bottleneck

Concept:
    This is your "before" script. Profile it with Nsight Systems and find
    why the GPU is idle most of the time.

    The workload simulates a perception model's inner loop:
      1. Get a batch (synthetic, generated on CPU as numpy)
      2. Move to GPU
      3. Run a small model
      4. Compute loss, backward, step

    The bottleneck planted here: the "data" is generated with slow numpy on
    every step, while the GPU sits idle waiting. Real-world equivalents:
    heavy CPU augmentations, .item() in the loop, non-pinned memory, etc.

How to profile:
    nsys profile -o baseline -t cuda,nvtx -f true python 01_profile_me.py
    nsys stats baseline.nsys-rep --report cuda_gpu_kern_sum
    nsys stats baseline.nsys-rep --report cuda_api_sum

What you should see in the GUI:
    - Long blank stripes on the GPU stream between kernel bursts.
    - Long blocks on the CPU thread in numpy code during those gaps.
    - Total kernel time will be a small fraction of wall time.

In the CLI summary:
    - `cuda_gpu_kern_sum` shows total kernel time (this is what the GPU
      actually did). Compare it to the script's wall clock. If kernel time
      is, say, 1.2 s and wall time is 5 s, the GPU was idle 76% of the time.
"""

import time
import numpy as np
import torch
import torch.nn as nn


def slow_numpy_batch(batch_size: int, dim: int) -> np.ndarray:
    # Simulate slow CPU data prep — e.g., point cloud loading + augmentation.
    # In your real voxelization pipeline, this would be the unbatched
    # Open3D / numpy preprocessing step.
    x = np.random.randn(batch_size, dim).astype(np.float32)
    # Some pointless work to make it slow:
    for _ in range(3):
        x = np.tanh(x) + 0.01 * np.sin(x * 3.14)
    return x


def main():
    assert torch.cuda.is_available()
    device = "cuda"

    batch_size, dim = 256, 4096
    model = nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    target = torch.randn(batch_size, dim, device=device)

    # Optional: tell nsys to only capture inside this region. Pair with
    #   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
    torch.cuda.profiler.start()

    t0 = time.perf_counter()
    n_steps = 50
    for step in range(n_steps):
        # ❌ CPU work on the critical path. GPU is idle during this.
        batch_np = slow_numpy_batch(batch_size, dim)

        # ❌ Synchronous H2D copy (no pin_memory, no non_blocking).
        x = torch.from_numpy(batch_np).to(device)

        y = model(x)
        loss = loss_fn(y, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ❌ .item() forces a GPU->CPU sync every step.
        _ = loss.item()

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    elapsed = time.perf_counter() - t0
    print(f"{n_steps} steps in {elapsed:.2f} s  ({elapsed / n_steps * 1000:.1f} ms/step)")
    print("Now run:")
    print("  nsys stats <report>.nsys-rep --report cuda_gpu_kern_sum")
    print("Compare total kernel time to wall time. The difference is GPU idle time.")


if __name__ == "__main__":
    main()

"""
03 — DataLoader bottleneck: the canonical "GPU is idle" finding

Concept:
    The single most common reason a real training run is slow is that the
    DataLoader can't feed the GPU fast enough. Symptoms in Nsight Systems:

      - Large gaps on the GPU compute stream
      - CPU threads (DataLoader workers) saturated
      - Total GPU utilization < 50% in nvidia-smi

This script lets you flip the optimizations on/off via env vars so you can
compare profiles side-by-side. Set them before invoking nsys:

    # BAD (default) — single-worker loader, no pinning, no prefetch:
    nsys profile -o bad -t cuda,nvtx -f true python 03_dataloader_idle.py

    # GOOD — multi-worker + pinned memory + non_blocking copies:
    GOOD=1 nsys profile -o good -t cuda,nvtx -f true python 03_dataloader_idle.py

What to compare in the two reports:
    1. Wall time of the loop (printed at the end).
    2. nsys stats --report cuda_gpu_kern_sum  → kernel time should be the SAME
       for both (the GPU is doing the same work).
    3. Gap analysis in the GUI: the BAD run has long blank stripes between
       kernels; the GOOD run has the copy stream busy in parallel with compute.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import nvtx


GOOD = os.environ.get("GOOD", "0") == "1"


class SyntheticDataset(Dataset):
    """A dataset that's intentionally slow on the CPU per item.
    Mimics: load a sample from disk + run python-side augmentation."""

    def __init__(self, n_items: int, dim: int):
        self.n_items = n_items
        self.dim = dim

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        # Slow CPU work per sample. With num_workers > 0 this happens in
        # parallel processes and overlaps with GPU compute.
        x = np.random.randn(self.dim).astype(np.float32)
        for _ in range(2):
            x = np.tanh(x) + 0.01 * np.sin(x * 3.14)
        return torch.from_numpy(x)


def main():
    assert torch.cuda.is_available()
    device = "cuda"
    dim = 4096
    batch_size = 256

    ds = SyntheticDataset(n_items=batch_size * 50, dim=dim)

    if GOOD:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,         # parallel CPU prep
            pin_memory=True,       # allocates host pages that can be DMA'd async
            persistent_workers=True,  # don't tear down workers every epoch
            prefetch_factor=4,     # how many batches each worker stages ahead
        )
        print("GOOD config: 4 workers, pin_memory, prefetch=4")
    else:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,         # everything happens in the main process
            pin_memory=False,
        )
        print("BAD config: 0 workers, no pinning")

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

    for step, batch in enumerate(loader):
        with nvtx.range(f"step_{step}"):
            with nvtx.range("h2d_copy"):
                # non_blocking=True only helps when the source is pinned memory.
                # The DMA engine can run the copy concurrently with the next
                # GPU kernel on a different stream.
                x = batch.to(device, non_blocking=GOOD)

            with nvtx.range("compute"):
                y = model(x)
                loss = loss_fn(y, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    elapsed = time.perf_counter() - t0
    print(f"{len(loader)} steps in {elapsed:.2f} s ({elapsed / len(loader) * 1000:.1f} ms/step)")


if __name__ == "__main__":
    main()

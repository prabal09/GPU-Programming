"""
02 — Inspecting the Triton code Inductor generated

Concept:
    Inductor's output is real Triton source code written to a temp dir.
    You can read it, copy it, learn from it. There are three ways to see it:

      A. Env var       : TORCH_LOGS="output_code"     (prints on stderr)
      B. In-code        : torch._logging.set_logs(output_code=True)
      C. Inspect cache : look in $TORCHINDUCTOR_CACHE_DIR (or /tmp/torchinductor_<user>/)

This script uses (B) so you don't have to remember the env var.

What to notice in the printed kernel:
    - `@triton.jit` decorator on the generated function.
    - `tl.program_id(0)` — the block index, like CUDA blockIdx.x.
    - `tl.arange(0, XBLOCK)` + `xoffset` — building per-thread offsets.
    - A `tl.load` for each input, ONE big block of math, ONE `tl.store`.
      That's the fusion — 5 pointwise ops became 1 kernel, 2 reads, 1 write.
    - The "XBLOCK" autotuning constant — Inductor picks block sizes.
"""

import torch
import torch._logging


def fused_pointwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    a = x * y
    b = a + x
    c = torch.sin(b)
    d = c * c
    return d + y


def main():
    assert torch.cuda.is_available()

    # Turn on Inductor's output_code log. This prints the generated Triton
    # source for every compiled graph to stderr. Other useful log topics:
    #   "graph_breaks"  — why Dynamo couldn't trace through your code
    #   "recompiles"    — why a function was recompiled
    #   "guards"        — what conditions the compiled artifact assumes
    #   "schedule"      — the kernel fusion schedule
    torch._logging.set_logs(output_code=True)

    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn_like(x)

    compiled = torch.compile(fused_pointwise)
    compiled(x, y)  # trigger compilation -> prints the Triton kernel.

    # If you want to find the file on disk later:
    #   import os
    #   print(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_<user>"))
    # Each compiled graph lives in a subdir; the generated kernel is `output_code.py`.


if __name__ == "__main__":
    main()

# Lesson 1: `torch.compile()` — from Python to Triton kernels

## What it actually does

When you wrap a function or `nn.Module` with `torch.compile`, PyTorch does *not* just JIT a faster Python interpreter. It runs a four-stage pipeline:

```
Python bytecode
   │
   ▼ TorchDynamo  (frame evaluation API; rewrites Python bytecode)
FX graph  (a static, traced representation of tensor ops)
   │
   ▼ AOTAutograd  (separates forward / backward, decomposes ops)
Lowered FX graph
   │
   ▼ TorchInductor  (the compiler backend)
Triton kernels  (auto-generated .py files with @triton.jit kernels)
   │
   ▼ Triton compiler → PTX → GPU
```

Key facts to internalize:

1. **TorchDynamo** intercepts your Python function the first time it runs. It walks the bytecode and builds an **FX graph** — basically a DAG of tensor ops. It does NOT execute Python during this trace; it specializes on input shapes/dtypes.
2. **Graph breaks** happen when Dynamo hits Python it can't trace (e.g. `tensor.item()`, `print`, data-dependent `if`). It splits the function into multiple compiled subgraphs with eager Python in between. Fewer breaks → bigger fused kernels → faster.
3. **TorchInductor** is the default backend. For CUDA tensors it **emits Triton code** — yes, the same Triton you're about to learn to write by hand. You can read what it generated.
4. **First call is slow** (compilation + autotuning). Subsequent calls hit the cache. Always warm up before timing.
5. **Recompilation** happens when input shapes/dtypes/devices change in ways Dynamo specialized on. Too much recompilation is the #1 perf footgun.

## Why this matters for your voxel/KNN kernel

Before writing a hand-rolled Triton kernel, run `torch.compile` on the PyTorch baseline. If Inductor already fuses it into one fast Triton kernel, your handwritten kernel needs to beat *that*, not eager PyTorch. The generated Triton is also a great starting reference — read it.

## Scripts in this folder

Run them in order. Each prints something instructive.

| File | What it teaches |
| --- | --- |
| `01_hello_compile.py` | The minimum example. Eager vs. compiled timing on a fused pointwise op. |
| `02_inspect_kernels.py` | How to dump the Triton code Inductor generated (`TORCH_LOGS=output_code`). |
| `03_graph_breaks.py` | What causes a graph break and how to see them (`TORCH_LOGS=graph_breaks`). |
| `04_modes_and_shapes.py` | `mode=` options + how dynamic shapes cause recompiles. |

## Setup on EC2

```bash
# On an Ubuntu 22.04 g4dn.xlarge / g5.xlarge etc. with NVIDIA drivers preinstalled (Deep Learning AMI)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You need **PyTorch ≥ 2.1** for a mature `torch.compile`. Triton ships with the PyTorch CUDA wheels, but installing it explicitly is harmless.

## How to read the rest

Each script has a `# Concept:` block at the top. Read that, run it, then read the code. If a script prints kernel source, **read the kernel** — that's where the learning is.

# GPU Programming — fundamentals

Working towards a hand-written Triton voxelization + batched KNN kernel. Before
writing kernels, this folder builds the two foundational skills:

1. **[01_torch_compile/](01_torch_compile/)** — understanding how PyTorch's own
   compiler turns Python into Triton kernels. The output of `torch.compile`
   is a real reference point: any handwritten Triton kernel needs to beat what
   Inductor already generates.
2. **[02_nsight_profiling/](02_nsight_profiling/)** — reading a GPU timeline.
   Knowing whether the GPU is starved (CPU/data bound) vs. genuinely compute
   bound is the difference between optimizing the right thing and chasing
   ghosts.

## Where to run

Both lessons assume an EC2 GPU instance (e.g. `g5.xlarge` / `g4dn.xlarge`) on
Ubuntu with NVIDIA drivers + CUDA 12.x. Either:

- Use the **AWS Deep Learning AMI (Ubuntu 22.04)** — PyTorch, CUDA, and Nsight
  Systems are pre-installed. Fastest path.
- Or vanilla Ubuntu 22.04: install drivers via `cuda-keyring`, then
  `pip install torch --index-url https://download.pytorch.org/whl/cu121` and
  `apt-get install nsight-systems-cli`.

## Order

```
01_torch_compile/01_hello_compile.py        # see the speedup
01_torch_compile/02_inspect_kernels.py      # read the generated Triton
01_torch_compile/03_graph_breaks.py         # debug compilation
01_torch_compile/04_modes_and_shapes.py     # modes + dynamic shapes

02_nsight_profiling/01_profile_me.py        # first profile — find the gaps
02_nsight_profiling/02_nvtx_annotated.py    # annotate the timeline
02_nsight_profiling/03_dataloader_idle.py   # bad vs. good DataLoader, side-by-side
```

Each folder has its own README with the conceptual overview. Read the README
first, run the scripts, then re-read with the actual output in front of you.

## After this

Next foundations to add (in roughly this order, suggested):

- `03_triton_basics/` — vector add, softmax, matmul. The classic Triton
  tutorials but with notes connecting concepts to what `torch.compile`
  produced in lesson 1.
- `04_memory_hierarchy/` — global vs. shared memory, coalescing, the
  HBM-vs-SRAM mental model.
- `05_voxel_knn/` — the actual target kernel.

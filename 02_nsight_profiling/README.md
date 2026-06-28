# Lesson 2: NVIDIA Nsight Systems — reading a GPU timeline

## What Nsight Systems (`nsys`) is, and isn't

| | Nsight Systems (`nsys`) | Nsight Compute (`ncu`) |
|---|---|---|
| Scope | **Whole system timeline.** CPU threads, GPU streams, CUDA API calls, kernel launches, memcpy, NVTX ranges. | **One kernel, deeply.** Memory throughput, occupancy, warp stalls, source-level counters. |
| Use for | "Is my GPU idle? Why?" "Is data loading the bottleneck?" "Are kernels overlapping?" | "Why is THIS Triton kernel slow? How do I make it faster?" |
| When | Most days, especially early in optimization. | After Systems tells you which kernel matters, you zoom in with Compute. |

You will spend ~90% of your profiling time in **Nsight Systems**. The single most useful thing it shows you is **GPU idle gaps** — periods where the GPU is doing nothing because Python / the DataLoader / a sync point is starving it.

## The mental model of the timeline

A typical Nsight Systems view stacks horizontal rows aligned in time:

```
   ┌─ time →
CPU thread 0 (Python)  ████░░██████░░░░████   ← Python code running
CPU thread 1 (worker)  ░░██████░░██████░░░░   ← DataLoader worker
CUDA API              ░ ▌ ▌ ▌  ▌▌▌▌  ▌ ▌ ▌    ← cudaLaunchKernel, cudaMemcpyAsync etc.
GPU stream 0          ░░░ ▓▓ ▓▓ ░░ ▓▓▓▓▓▓ ░   ← actual kernels executing
GPU stream 1 (copy)   ░ ▒▒ ░░░░░░░░░░ ▒▒ ░░   ← H2D / D2H copies (if pinned + non_blocking)
NVTX ranges            [ forward ][ back ][step]  ← your annotations
```

Things to look for, in priority order:

1. **Gaps on "GPU stream 0"** = GPU is idle. Find what's running on CPU during those gaps. That's your bottleneck.
2. **Long CUDA API calls** (especially `cudaMemcpy*` without "Async" or `cudaStreamSynchronize`) = an unintended host↔device sync. Common cause: `.item()`, `print(tensor)`, `tensor.cpu()`, `len()` on a tensor whose shape isn't static.
3. **No overlap between "GPU compute" and "GPU copy" streams** = your DataLoader isn't using `pin_memory=True` + `non_blocking=True`. You're paying full transfer cost serially.
4. **Many tiny kernels** = launch overhead dominates. Candidates for `torch.compile` (fusion) or `mode="reduce-overhead"` (CUDA Graphs).
5. **NVTX ranges showing forward/backward/step** — read your own labels; they orient you in the timeline.

## Installing Nsight Systems on EC2 (Ubuntu)

The Deep Learning AMI usually has it. If not:

```bash
# Add NVIDIA's repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y nsight-systems-cli

nsys --version   # should print 2024.x or newer
```

CLI-only is fine on a headless EC2 box. You'll **transfer the .nsys-rep file** to your laptop and open it in the **Nsight Systems GUI** (free download: https://developer.nvidia.com/nsight-systems). The GUI is where the timeline visualization lives.

## The five commands you actually use

```bash
# 1. Profile a Python script end-to-end. Writes profile.nsys-rep.
nsys profile -o profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
    python my_script.py

# Quick version (profiles everything from start):
nsys profile -o profile -t cuda,nvtx python my_script.py

# 2. Just print the kernel summary to stdout (no GUI needed for first pass).
nsys stats profile.nsys-rep --report cuda_gpu_kern_sum

# 3. Memory transfer summary.
nsys stats profile.nsys-rep --report cuda_gpu_mem_size_sum

# 4. Top CPU-side CUDA API calls — useful to find slow syncs.
nsys stats profile.nsys-rep --report cuda_api_sum

# 5. Open the report GUI on your laptop:
#    (copy profile.nsys-rep to your laptop, then File -> Open in Nsight Systems)
```

Flags worth knowing:

| Flag | Meaning |
|---|---|
| `-t cuda,nvtx,osrt` | Trace CUDA API, NVTX ranges, OS runtime. Add `python-gil` to see GIL contention. |
| `-o name` | Output file (creates `name.nsys-rep`). |
| `--capture-range=cudaProfilerApi` | Only capture between `torch.cuda.profiler.start()` and `.stop()`. Use to skip warmup / dataloader init. |
| `--force-overwrite=true` | Allow overwriting an existing report. |
| `--stats=true` | Print summary stats after profiling finishes (skip step 2). |

## Scripts in this folder

| File | What it teaches |
| --- | --- |
| `01_profile_me.py` | A baseline training loop with an intentional CPU-side bottleneck. Profile this first. |
| `02_nvtx_annotated.py` | Same workload but with NVTX ranges so the timeline is readable. |
| `03_dataloader_idle.py` | A DataLoader that starves the GPU. Classic finding: GPU compute < 30% utilization. |

For each script, the workflow is:

```bash
nsys profile -o run -t cuda,nvtx -f true python 01_profile_me.py
nsys stats run.nsys-rep --report cuda_gpu_kern_sum
scp run.nsys-rep <laptop>:    # then open in the GUI
```

## What "good" looks like on the timeline

A well-optimized step:
- GPU stream is a **dense block of color** — no gaps.
- CPU is mostly idle / just queueing the next batch — the GPU is the bottleneck (this is desirable!).
- Copy stream overlaps with compute stream during prefetch.
- `cudaMemcpyAsync` events are tiny and async; no `cudaDeviceSynchronize` in your training loop.

If you see the opposite — long gaps on GPU, CPU thread saturated — you're CPU-bound, and `torch.compile` won't save you. You need to fix the data path (faster DataLoader, more workers, pin_memory, prefetching, smaller transforms, GPU-side augmentation).

"""
03 — Graph breaks: the single most important debugging skill

Concept:
    TorchDynamo can't trace through arbitrary Python. When it hits something
    untraceable, it ENDS the current subgraph, runs that subgraph compiled,
    runs the untraceable Python in eager mode, then starts a NEW subgraph.

    Each break = an extra GPU sync, an extra kernel launch, lost fusion
    opportunity. A "fully compiled" function has zero breaks.

Common causes:
    1. .item() / .tolist() / float(tensor)  -> forces a GPU->CPU sync to read a value
    2. print(tensor)                         -> same
    3. Data-dependent control flow (if tensor.sum() > 0: ...)
    4. Calls into C extensions or unsupported libs (cv2, np operations on cuda tensors via .cpu(), etc.)
    5. In-place ops on inputs that Dynamo can't prove safe
    6. assert with tensor condition

How to see them:
    - Run with env var:  TORCH_LOGS=graph_breaks python 03_graph_breaks.py
    - Or use fullgraph=True (this script does that) so Dynamo RAISES on a break
      instead of silently splitting. Best mode for development.
"""

import torch


def bad_function(x: torch.Tensor) -> torch.Tensor:
    # Dynamo can trace the math:
    y = x * 2 + 1

    # ...but this forces a GPU->CPU sync. Dynamo cannot fold this into the graph.
    # In default mode it's a graph break. With fullgraph=True it RAISES.
    threshold = y.sum().item()

    # Python control flow that depends on a CPU value derived from a tensor.
    if threshold > 0:
        return y * 3
    else:
        return y * -3


def good_function(x: torch.Tensor) -> torch.Tensor:
    # Same intent, no break. torch.where stays on the GPU as a single op
    # and Dynamo can trace it cleanly.
    y = x * 2 + 1
    return torch.where(y.sum() > 0, y * 3, y * -3)


def main():
    assert torch.cuda.is_available()
    x = torch.randn(1 << 16, device="cuda")

    # fullgraph=True is the strict mode. Use it during development —
    # it converts silent perf losses into loud exceptions you can fix.
    print("=== bad_function with fullgraph=True ===")
    try:
        torch.compile(bad_function, fullgraph=True)(x)
    except Exception as e:
        # The exception message tells you EXACTLY which line broke the graph.
        print(f"Graph break caught:\n{type(e).__name__}: {str(e)[:400]}...\n")

    print("=== good_function with fullgraph=True ===")
    out = torch.compile(good_function, fullgraph=True)(x)
    print(f"Compiled without breaks. Output shape: {out.shape}\n")

    # Bonus: you can also use torch._dynamo.explain to get a structured report
    # WITHOUT running the function, listing every break and its cause.
    print("=== torch._dynamo.explain on bad_function ===")
    explanation = torch._dynamo.explain(bad_function)(x)
    print(explanation)


if __name__ == "__main__":
    main()

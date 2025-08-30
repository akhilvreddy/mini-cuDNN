# mini-cuDNN
A toy deep learning GPU backend written in CUDA for exploring what NVIDIA’s cuDNN does under the hood.

```txt
mini-cudnn/
├─ mydnn/
│  ├─ conv.cu          # core CUDA kernels
│  ├─ conv.cpp         # C++ launcher + bindings
├─ tests/
│  └─ test_conv.py     # our kernel vs torch
├─ setup.py            # build script using torch CUDAExtension (building the kernel)
├─ README.md
└─ .gitignore
```

---

The motivation for this project came from me wanting the understand what is going on underneath the PyTorch API, especially when it is run on an NVIDIA GPU.

Whenever we write something like
```py
y = torch.nn.Conv2d(3, 64, kernel_size=3)(x)
```

on the GPU, it calls cuDNN under the hood and for the rest of the computation, we treat cuDNN and it's functionality as a black box. 

Specifically it uses 2 different tools depending on the tensor op: 
- **cuDNN** (for conv, pooling, etc.)  
- **cuBLAS** (for GEMM/matmul)

Since almost every single op is some variant of a matmul, cuDNN actually will call into cuBLAS to crunch these operations.

## How does cuDNN help? 

At it's core, cuDNN is a closed source lib of NVIDIA GPU-optimized kernels. It decides many things like
- The algorithm to use for conv (naive, im2col + GEMM, FFT, Winograd) → each has their own pro/con based on the input
- How to tile data in shared memory
- How to schedule threads/warps for max GPU utilization
- How to mix precision for max throughput

cuDNN is essentially the brain behind crunching the tensors as quick as possible. cuDNN kernels are just CUDA programs so they can expose CUDA's API which the GPU can execute in parallel

```txt
          ┌─────────────────┐
          │ Input Tensor x  │   shape [N, C, H, W]
          │    (PyTorch)    │
          └────────┬────────┘
                   │
                   ▼
        ┌───────────────────────┐
        │ im2col (lowering op)  │
        │   - unfold patches    │
        │   - create big 2D mat │
        │       (cuDNN)         │
        └────────┬──────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ GEMM (matrix multiply)│
        │   [patches × CRS]  ×  │
        │   [CRS × K]           │
        │      (cuBLAS)         │
        └────────┬──────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Output Tensor y │   shape [N, K, Ho, Wo]
        │    (PyTorch)    │
        └─────────────────┘
```

Here's an ASCII image that I got GPT to draw for me. This shows the clear handoff: PyTorch (input) → cuDNN (im2col transform / some other conv algos) → cuBLAS (heavy matmul) → PyTorch (output).

## Scope of this project

I wanted to start super simple and get an in house solution working for PyTorch's `torch.nn.functional.conv2d`. I could get that working by doing something like

```py
y = mydnn.conv2d_naive(x, w, stride_h=1, stride_w=1, pad_h=0, pad_w=0)
```

so the interface would mimic PyTorch style calls but would run my own kernel under the hood. To see how well this in-house kernel is doing we can benchmark it against PyTorch's native call.

---

All the experiment code is written in `main.ipynb` (check there for results!)
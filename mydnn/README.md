Here's the flow between these files: 

```txt
Python call:
    y = mydnn.conv2d_naive(x, w, 1, 1, 1, 1)
       │
       ▼
C++ binding (conv.cpp):
    conv2d_naive(...)   → checks tensors, calls conv2d_naive_cuda(...)
       │
       ▼
CUDA launcher (conv.cu):
    conv2d_naive_cuda(...)   → launches conv2d_naive_kernel<<<blocks,threads>>>()
       │
       ▼
GPU kernel (conv.cu):
    conv2d_naive_kernel(...) → actual math
```


I am going to use Google Colab's cloud Tesla T4 and CUDA running on that so I will import this kernel into that notebook and then will call `mydnn.conv2d_naive` from there for testing.
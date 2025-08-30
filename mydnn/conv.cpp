#include <torch/extension.h>

// forward-declare the CUDA launcher from conv.cu
torch::Tensor conv2d_naive_cuda(
    torch::Tensor x, torch::Tensor w,
    int stride_h, int stride_w,
    int pad_h, int pad_w);

// C++ wrapper that PyTorch will call
torch::Tensor conv2d_naive(
    torch::Tensor x, torch::Tensor w,
    int stride_h = 1, int stride_w = 1,
    int pad_h = 0, int pad_w = 0
) {
    // sanity checks
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "weights must be float32");

    // call down into the CUDA implementation
    return conv2d_naive_cuda(
        x.contiguous(), w.contiguous(),
        stride_h, stride_w, pad_h, pad_w
    );
}

// bind this wrapper to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_naive", &conv2d_naive, "Naive Conv2d (CUDA)");
}
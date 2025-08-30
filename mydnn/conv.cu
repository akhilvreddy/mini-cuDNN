#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// naive convolution kernel, one thread computes one output element
// this core kernel needs to be wrapped so that it can be called (function below does that)
__global__ void conv2d_naive_kernel(
    const float* __restrict__ x,   // [N,C,H,W]
    const float* __restrict__ w,   // [K,C,R,S]
    float* __restrict__ y,         // [N,K,Ho,Wo]
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int Ho, int Wo
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * Ho * Wo;
    if (idx >= total) return;

    int wo = idx % Wo;
    int ho = (idx / Wo) % Ho;
    int k  = (idx / (Wo * Ho)) % K;
    int n  = idx / (Wo * Ho * K);

    float acc = 0.0f;

    int h_in0 = ho * stride_h - pad_h;
    int w_in0 = wo * stride_w - pad_w;

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int h_in = h_in0 + r;
            if (h_in < 0 || h_in >= H) continue;
            for (int s = 0; s < S; ++s) {
                int w_in = w_in0 + s;
                if (w_in < 0 || w_in >= W) continue;

                int x_idx = ((n * C + c) * H + h_in) * W + w_in;
                int w_idx = ((k * C + c) * R + r) * S + s;

                acc += x[x_idx] * w[w_idx];
            }
        }
    }

    int y_idx = ((n * K + k) * Ho + ho) * Wo + wo;
    y[y_idx] = acc;
}

// C++ launcher called from conv.cpp
torch::Tensor conv2d_naive_cuda(
    torch::Tensor x, torch::Tensor w,
    int stride_h, int stride_w, int pad_h, int pad_w
) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int K = w.size(0), R = w.size(2), S = w.size(3);

    int Ho = (H + 2 * pad_h - R) / stride_h + 1;
    int Wo = (W + 2 * pad_w - S) / stride_w + 1;

    auto y = torch::empty({N, K, Ho, Wo}, x.options());

    int total = N * K * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_naive_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        K, R, S,
        stride_h, stride_w,
        pad_h, pad_w,
        Ho, Wo
    );

    return y;
}
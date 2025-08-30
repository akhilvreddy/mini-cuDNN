# to run
# bash: pytest -q

import os
import math
import pytest
import torch
import torch.nn.functional as F

import mydnn # built from local extension

# If you want to ensure we compare against pure PyTorch kernels (and not cuDNN autotuned oddities),
# you can uncomment the next lines:

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def _rand_tensor(shape, seed=0, device="cuda"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(*shape, device=device, dtype=torch.float32, generator=g)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("N,C,H,W", [(1, 1, 8, 8), (2, 3, 16, 17), (4, 8, 15, 13)])
@pytest.mark.parametrize("K,R,S", [(1, 3, 3), (4, 3, 3), (8, 5, 5)])
@pytest.mark.parametrize("stride_h,stride_w,pad_h,pad_w", [(1,1,0,0), (1,1,1,1), (2,2,1,1)])
def test_conv2d_naive_matches_torch(N, C, H, W, K, R, S, stride_h, stride_w, pad_h, pad_w):
    # generate random inputs
    x = _rand_tensor((N, C, H, W), seed=123)
    w = _rand_tensor((K, C, R, S), seed=456)

    # reference
    y_ref = F.conv2d(x, w, bias=None, stride=(stride_h, stride_w), padding=(pad_h, pad_w))

    # under test
    y = mydnn.conv2d_naive(x, w, stride_h, stride_w, pad_h, pad_w)

    # compare
    max_abs = (y - y_ref).abs().max().item()
    max_rel = ( (y - y_ref).abs() / (y_ref.abs() + 1e-6) ).max().item()
    assert max_abs < 1e-4 and max_rel < 1e-4, f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_dtype_raises():
    x = torch.randn(1, 1, 8, 8, device="cuda", dtype=torch.float16)
    w = torch.randn(1, 1, 3, 3, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = mydnn.conv2d_naive(x, w, 1, 1, 0, 0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_basic_shape_computation():
    N,C,H,W = 2,3,16,16
    K,R,S = 4,3,3
    sh,sw,ph,pw = 2,2,1,1
    x = torch.randn(N,C,H,W, device="cuda")
    w = torch.randn(K,C,R,S, device="cuda")
    y = mydnn.conv2d_naive(x, w, sh, sw, ph, pw)
    Ho = (H + 2*ph - R)//sh + 1
    Wo = (W + 2*pw - S)//sw + 1
    assert y.shape == (N, K, Ho, Wo)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_small_known_case():
    # 1x1x3x3 input, 1x1x3x3 filter, no pad/stride
    x = torch.tensor([[[[1.,2.,3.],
                        [4.,5.,6.],
                        [7.,8.,9.]]]], device="cuda")
    w = torch.tensor([[[[1.,0.,-1.],
                        [1.,0.,-1.],
                        [1.,0.,-1.]]]], device="cuda")
    y_ref = F.conv2d(x, w, bias=None, stride=1, padding=0)
    y = mydnn.conv2d_naive(x, w, 1, 1, 0, 0)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6)

# optional quick benchmark (run with: pytest -q -k bench -s)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_naive_conv2d():
    N,C,H,W = 8,64,128,128
    K,R,S = 64,3,3
    sh,sw,ph,pw = 1,1,1,1
    x = torch.randn(N,C,H,W, device="cuda")
    w = torch.randn(K,C,R,S, device="cuda")

    # Warmup
    for _ in range(5):
        _ = mydnn.conv2d_naive(x, w, sh, sw, ph, pw)
    torch.cuda.synchronize()

    import time
    iters = 10
    t0 = time.time()
    for _ in range(iters):
        _ = mydnn.conv2d_naive(x, w, sh, sw, ph, pw)
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000.0 / iters
    print(f"naive conv2d avg: {dt:.2f} ms  (N={N},C={C},H={H},W={W},K={K},R={R},S={S})")
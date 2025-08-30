# to build
# bash: pip install -e .

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mydnn",
    ext_modules=[
        CUDAExtension(
            name="mydnn",
            sources=["mydnn/conv.cpp", "mydnn/conv.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
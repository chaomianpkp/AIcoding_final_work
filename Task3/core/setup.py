import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def gather_paths():
    sources = [
        "src/pybind_module.cpp",
        "src/tensor.cu",
        "src/layers.cu",
        "src/activations.cu",
    ]
    include_dirs = [
        "include",
    ]
    return sources, include_dirs


sources, include_dirs = gather_paths()

setup(
    name="mytensor",
    version="0.1.0",
    description="Pybind11 bindings for custom CUDA Tensor and NN operators",
    packages=find_packages(),
    package_dir={"mytensor": "mytensor"},
    python_requires=">=3.8",
    ext_modules=[
        CUDAExtension(
            name="mytensor._C",
            sources=sources,
            include_dirs=include_dirs,
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17", "-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)


import os
import sys
from setuptools import setup, find_packages


def _is_sdist_build():
    """Check if we are building a source distribution (no compilation needed)."""
    return "sdist" in sys.argv or "egg_info" in sys.argv


def get_cuda_arch_flags():
    import torch

    cuda_arch = os.environ.get("CUDA_ARCH")
    if cuda_arch:
        return [f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to build mhc (set CUDA_ARCH env var for headless builds)"
        )
    major, minor = torch.cuda.get_device_capability()
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def get_extra_defines():
    import torch

    cuda_arch = os.environ.get("CUDA_ARCH")
    if cuda_arch:
        defines = []
        if int(cuda_arch) >= 90:
            defines.append("-DMHC_ENABLE_PDL")
        return defines
    if not torch.cuda.is_available():
        return []
    major, _ = torch.cuda.get_device_capability()
    defines = []
    if major >= 9:
        defines.append("-DMHC_ENABLE_PDL")
    return defines


# Skip CUDA extension entirely for sdist builds (no CUDA toolkit needed)
if _is_sdist_build():
    ext_modules = []
    cmdclass = {}
else:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="mhc_cuda",
            sources=["src/python/bindings.cu"],
            include_dirs=[
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "src/csrc/include"
                ),
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "src/csrc/kernels"
                ),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--use_fast_math",
                ]
                + get_cuda_arch_flags()
                + get_extra_defines(),
            },
            libraries=["cublas", "cublasLt"],
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="mhc-cuda",
    version="0.1.0",
    description="CUDA implementation of Manifold-Constrained Hyper-Connections",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Andre Slavescu",
    url="https://github.com/AndreSlavescu/mHC.cu",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[
        "accelerate>=0.26.0",
        "datasets",
        "safetensors",
        "sentencepiece",
        "tiktoken",
        "torch>=2.0.0",
        "tqdm",
        "transformers",
    ],
    extras_require={
        "dev": ["black", "pytest", "ruff"],
    },
)

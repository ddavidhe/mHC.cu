import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import modal

APP_NAME = "mhc-cu"
HF_CACHE_VOLUME = "mhc-hf-cache"
RUNS_VOLUME = "mhc-runs"

app = modal.App(APP_NAME)

_base = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "python3-dev", "python3-pip"
    )
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/bin/python",
        "python3 -m pip install numpy ninja pytest",
        "python3 -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
        "python3 -m pip install datasets sentencepiece tiktoken tqdm transformers accelerate>=0.26.0",
    )
)

project_root = Path(__file__).parent
ignore_patterns = [
    ".git",
    ".stamps",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "runs",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.o",
]
_BUILD_ENV = "CC=gcc CXX=g++ CUDACXX=/usr/local/cuda/bin/nvcc PIP_NO_BUILD_ISOLATION=1"

h100_image = _base.add_local_dir(
    project_root, remote_path="/workspace", copy=True, ignore=ignore_patterns
).run_commands(
    f"cd /workspace && {_BUILD_ENV} CUDA_ARCH=90 make benchgen build install"
)
b200_image = _base.add_local_dir(
    project_root, remote_path="/workspace", copy=True, ignore=ignore_patterns
).run_commands(
    f"cd /workspace && {_BUILD_ENV} CUDA_ARCH=100 make benchgen build install"
)

runs_volume = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


def _print_env() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        return
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def _run_make(target: str) -> None:
    env = os.environ.copy()
    env.setdefault("CC", "gcc")
    env.setdefault("CXX", "g++")
    env.setdefault("CUDACXX", "/usr/local/cuda/bin/nvcc")
    env.setdefault("PIP_NO_BUILD_ISOLATION", "1")
    print(f"Running: make {target}")
    subprocess.run(["make", target], check=True, cwd="/workspace", env=env)


def _run_tests(scope: str) -> None:
    if scope in ("native", "all"):
        _run_make("test")
    if scope in ("python", "all"):
        _run_make("test-python")


def _run_benchmarks(scope: str) -> None:
    if scope in ("native", "all"):
        subprocess.run(
            'for b in build/bench_*; do echo "Running $b..."; $b; done',
            shell=True,
            check=True,
            cwd="/workspace",
        )
    if scope in ("python", "all"):
        _run_make("bench-python")


def _run_job(mode: str, scope: str) -> None:
    sys.path.insert(0, "/workspace")
    _print_env()
    if mode == "test":
        _run_tests(scope)
        return
    if mode == "bench":
        _run_benchmarks(scope)
        return
    if mode == "all":
        _run_tests(scope)
        _run_benchmarks(scope)
        return
    raise ValueError("mode must be 'test', 'bench', or 'all'")


def _run_train(train_args: Optional[str]) -> Tuple[str, str]:
    sys.path.insert(0, "/workspace/src/python")
    _print_env()
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("MHC_RUNS_ROOT", "/runs")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import mhc.trainer as trainer

    args = shlex.split(train_args) if train_args else []
    if "--runs-root" not in args:
        args += ["--runs-root", "/runs"]
    run_name, out_dir = trainer.run_training(args)
    # Flush volume writes so they're visible to `modal volume get` immediately
    runs_volume.commit()
    return run_name, out_dir


@app.function(gpu="H100", image=h100_image, timeout=3600)
def run_h100(mode: str, scope: str) -> None:
    _run_job(mode, scope)


@app.function(gpu="B200", image=b200_image, timeout=3600)
def run_b200(mode: str, scope: str) -> None:
    _run_job(mode, scope)


@app.function(
    gpu="H100",
    image=h100_image,
    volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache_volume},
    timeout=12 * 3600,
)
def train_h100(train_args: Optional[str]) -> Tuple[str, str]:
    return _run_train(train_args)


@app.function(
    gpu="B200",
    image=b200_image,
    volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache_volume},
    timeout=12 * 3600,
)
def train_b200(train_args: Optional[str]) -> Tuple[str, str]:
    return _run_train(train_args)


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    mode: str = "test",
    scope: str = "all",
    train_args: Optional[str] = None,
    download: bool = False,
) -> None:
    gpu_norm = gpu.strip().lower()
    mode_norm = mode.strip().lower()
    scope_norm = scope.strip().lower()
    if gpu_norm not in ("h100", "b200"):
        raise ValueError("gpu must be 'h100' or 'b200'")
    run_fn = run_h100 if gpu_norm == "h100" else run_b200
    train_fn = train_h100 if gpu_norm == "h100" else train_b200
    if mode_norm == "train":
        run_name, _ = train_fn.remote(train_args)
        if download:
            _download_run(run_name)
        return
    if scope_norm not in ("native", "python", "all"):
        raise ValueError("scope must be 'native', 'python', or 'all'")
    run_fn.remote(mode_norm, scope_norm)


def _download_run(run_name: str) -> None:
    if not run_name:
        return
    run_name = run_name.strip()
    local_root = Path("runs")
    local_root.mkdir(exist_ok=True)
    local_dir = local_root / run_name
    if local_dir.exists():
        shutil.rmtree(local_dir)
    # Download into parent dir — modal volume get preserves the directory name
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            RUNS_VOLUME,
            f"/{run_name}",
            str(local_root),
        ],
        check=True,
    )

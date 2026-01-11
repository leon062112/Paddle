# 2025.10.30
# 绑定 Attention 到 Paddle 框架
# 使用方法: CUDA_ARCH=89 python setup.py install
# 编译标志中 sm 的型号可能需要依据平台而改变  A100:sm80 | 4080:sm89 | 3090:sm86
import os
from pathlib import Path
from site import getsitepackages
import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup
from paddle.utils.cpp_extension.extension_utils import _get_all_paddle_includes_from_include_root

current_path = os.getcwd()

# 从环境变量获取 CUDA 架构
cuda_arch = os.environ.get("CUDA_ARCH", "80")
print(f" >>> [Binding INFO] Compiling for CUDA architecture: sm_{cuda_arch}")

# 获取 Paddle 头文件路径
paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_include_dir = Path(site_packages_path) / "paddle/include"
    if paddle_include_dir.exists():
        paddle_includes.extend(
            _get_all_paddle_includes_from_include_root(paddle_include_dir)
        )

# 添加 Eigen 头文件路径（避免 device_context.h 找不到 unsupported/Eigen/CXX11/Tensor）
# 可通过环境变量 EIGEN3_INCLUDE_DIR 覆盖
eigen_candidates = []
env_eigen = os.environ.get("EIGEN3_INCLUDE_DIR")
if env_eigen:
    eigen_candidates.append(Path(env_eigen))
eigen_candidates.extend([
    Path("/usr/include/eigen3"),
    Path("/usr/local/include/eigen3"),
    Path("/opt/homebrew/include/eigen3"),  # macOS (可选)
])

for p in eigen_candidates:
    if (p / "unsupported/Eigen/CXX11/Tensor").exists():
        paddle_includes.append(str(p))
        print(f" >>> [Binding INFO] Using Eigen include dir: {p}")
        break

# 添加当前项目的头文件路径
paddle_includes.append(f"{current_path}/src/include")
paddle_includes.append(f"{current_path}/src/include/cutlass/include")

# 编译参数
extra_compile_args = {
    "cxx": ["-O3", "-w"],
    "nvcc": [
        "-O3", "-w",
        f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"
    ],
}

# 源文件
sources = ["src/custom_attention.cc", "src/custom_attention_cuda.cu"]

setup(
    name='custom_attention',
    ext_modules=CUDAExtension(
        sources=sources,
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args,
        verbose=True,
    ),
)
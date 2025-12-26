# 2025.10.30 
# 绑定 Attention 到 Paddle 框架
#
# python setup.py 89
# 编译标志中 sm 的型号可能需要依据平台而改变  A100:sm80 | 4080:sm89 | 3090:sm86
import os
import sys
import glob
from setuptools import find_packages, setup
from paddle.utils.cpp_extension import CUDAExtension
import multiprocessing

current_path = os.getcwd()

if len(sys.argv) > 1:
    cuda_arch = sys.argv.pop(1)  # 取出第一个参数
    # cuda_arch = 89  # 取出第一个参数
    try:
        int(cuda_arch)  # 确保输入的是整数
    except ValueError:
        print("Error: CUDA architecture must be a number (e.g., 80, 89).")
        sys.exit(1)
else:
    cuda_arch = "80"  # 默认架构

print(f" >>> [Binding INFO] Compiling for CUDA architecture: sm_{cuda_arch}")

# 并行编译设置
num_cores = multiprocessing.cpu_count()
max_jobs = min(32, num_cores * 2)  # 限制最大作业数避免内存溢出
print(f" >>> [Binding INFO] Using parallel compilation with {max_jobs} jobs")

extra_compile_args = {
    "nvcc" : ["-O3", "-w", 
    "-I/usr/local/cuda/include", 
    f"-I{str(current_path)}/ops/src/include/",                 # 添加自己的 依赖路径
    f"-I{str(current_path)}/ops/src/include/cutlass/include",  # 添加 cutlass 依赖
    f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"],  # 此处gencode后面的等号不可以省
}
extra_link_args = []


def get_extensions():    
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "ops", "src")
    
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*_cuda.cu")))
    
    # 从文件名中提取算子名称
    extension_names = set()
    for source in sources + cuda_sources:
        base_name = os.path.basename(source)
        name, _ = os.path.splitext(base_name)
        if name.endswith("_cuda"):
            name = name[:-5]  # 去掉 "_cuda" 后缀
        extension_names.add(name)
            
    ext_modules = []
    for name in extension_names:
        # 动态生成每个扩展的源文件列表
        extension_sources = [s for s in sources + cuda_sources if name in os.path.basename(s)]
        ext_modules.append(
            CUDAExtension(
                name=name,
                sources=extension_sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        )
    
    return ext_modules

setup(
    name='Custom_Attention',     
    packages=find_packages(),
    version='0.1.1',
    author='Pinecone Fund Custom Attention',
    
    ext_modules=get_extensions(),
    
    install_requires=["torch"],
    description="customed operator linked with cutlass implement",
    
    # # cmdclass={'build_ext': BuildExtension}
    # cmdclass={
    #     'build_ext': BuildExtension.with_options(
    #         use_ninja=True,  # 强制使用ninja
    #         max_jobs=max_jobs,  # 并行任务数
    #         no_python_abi_suffix=True  # 简化输出文件名
    #     )
    # }
)
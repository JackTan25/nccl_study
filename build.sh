#!/bin/bash
# NCCL Study 项目构建脚本

set -e

# 激活 MPI 环境
eval "$(conda shell.bash hook)"
conda activate /home/tanboyu.tby/mpi_env

# 设置环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/home/tanboyu.tby/mpi_env/lib:$LD_LIBRARY_PATH

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

# 创建构建目录
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 运行 CMake
echo "=== Configuring with CMake ==="
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DMPI_CXX_COMPILER=/home/tanboyu.tby/mpi_env/bin/mpicxx

# 编译
echo ""
echo "=== Building ==="
cmake --build . -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: ${BUILD_DIR}/demo"
echo ""
echo "To run: cd ${BUILD_DIR} && make run"
echo "Or manually: mpirun -np 2 --allow-run-as-root ${BUILD_DIR}/demo"


# NCCL Study

NVIDIA Collective Communications Library (NCCL) 学习项目，演示多 GPU 之间的集合通信操作。

## 环境要求

- CUDA 12.x
- OpenMPI
- NCCL (通常随 CUDA 一起安装)
- CMake 3.18+

## 快速开始

### 1. 首次设置环境

如果系统没有 MPI，需要先创建 conda 环境：

```bash
conda create -y -p ~/mpi_env python=3.10 openmpi cmake -c conda-forge
```

### 2. 编译项目

```bash
./build.sh
```

### 3. 运行示例

```bash
cd build && make run
```

或者手动运行：

```bash
conda activate ~/mpi_env
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:~/mpi_env/lib:$LD_LIBRARY_PATH
mpirun -np 2 --allow-run-as-root ./build/demo
```

## 项目结构

```
nccl_study/
├── CMakeLists.txt          # CMake 构建配置
├── build.sh                # 一键构建脚本
├── README.md               # 本文件
├── build/                  # 构建输出目录 (自动生成)
└── nccl_chap1/             # 第一章：NCCL 基础
    └── demo.cpp            # MPI + NCCL AllReduce 示例
```

## 示例说明

### nccl_chap1/demo.cpp

这个示例演示了如何结合 MPI 和 NCCL 进行多 GPU AllReduce 操作：

- 使用 MPI 进行多进程管理
- 每个 MPI 进程管理 2 个 GPU
- 使用 NCCL 进行跨 GPU 的 AllReduce 通信

**关键概念：**

1. **ncclGetUniqueId** - 生成唯一 ID 用于初始化通信器
2. **ncclCommInitRank** - 初始化 NCCL 通信器
3. **ncclAllReduce** - 执行 AllReduce 操作（所有 GPU 数据求和并广播）
4. **ncclGroupStart/End** - 将多个 NCCL 操作组合在一起

## 常用命令

| 命令 | 说明 |
|------|------|
| `./build.sh` | 编译项目 |
| `cd build && make run` | 运行 demo（2 进程，4 GPU）|
| `rm -rf build && ./build.sh` | 清理并重新编译 |
| `mpirun -np N ./build/demo` | 使用 N 个进程运行（需要 2N 个 GPU）|

## 环境变量

运行前需要设置的环境变量：

```bash
# 激活 MPI 环境
eval "$(conda shell.bash hook)"
conda activate ~/mpi_env

# 设置库路径
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:~/mpi_env/lib:$LD_LIBRARY_PATH
```

## 参考资料

- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL GitHub](https://github.com/NVIDIA/nccl)
- [OpenMPI 文档](https://www.open-mpi.org/doc/)


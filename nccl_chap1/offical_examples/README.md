# NCCL 官方示例

这些示例代码来自 NVIDIA NCCL 官方文档：

📖 **官方文档**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html

## 示例列表

| 示例 | 说明 | 需要 MPI | GPU 数量 |
|------|------|----------|----------|
| example1 | 单进程、单线程、多设备 | ❌ | 4 |
| example2 | 每进程一个设备 | ✅ | N (进程数) |
| example3 | 每线程多个设备 | ✅ | 2×N |
| example4 | 每设备多个通信器 | ✅ | N |

## 编译

在项目根目录运行：

```bash
./build.sh
```

## 运行示例

### Example 1: 单进程、单线程、多设备

```bash
cd build && make run_example1
# 或直接运行
./build/example1
```

**特点**：
- 单个进程管理所有 4 个 GPU
- 使用 `ncclCommInitAll()` 一次性初始化所有通信器
- 使用 Group API 管理多 GPU 操作

### Example 2: 每进程一个设备

```bash
cd build && make run_example2
# 或手动指定进程数
mpirun -np 4 --allow-run-as-root ./build/example2
```

**特点**：
- 每个 MPI 进程使用 1 个 GPU
- 使用 `ncclCommInitRank()` 初始化通信器
- 通过 `localRank` 自动选择 GPU

### Example 3: 每线程多个设备

```bash
cd build && make run_example3
# 或
mpirun -np 2 --allow-run-as-root ./build/example3
```

**特点**：
- 每个 MPI 进程使用 2 个 GPU
- 使用 Group API 管理多 GPU
- 需要 `ncclGroupStart()` / `ncclGroupEnd()` 包裹操作

### Example 4: 每设备多个通信器

```bash
cd build && make run_example4
# 或
mpirun -np 4 --allow-run-as-root ./build/example4
```

**特点**：
- 每个设备创建多个通信器
- 演示阻塞 vs 非阻塞通信器初始化
- 使用 `ncclCommInitRankConfig()` 创建非阻塞通信器

## 关键 API 对比

| 场景 | 初始化 API | 是否需要 Group API |
|------|------------|-------------------|
| 单进程多 GPU | `ncclCommInitAll()` | 是 |
| 多进程单 GPU | `ncclCommInitRank()` | 否 |
| 多进程多 GPU | `ncclCommInitRank()` | 是 |
| 非阻塞初始化 | `ncclCommInitRankConfig()` | 视情况 |

## 代码结构

```
offical_examples/
├── README.md       # 本文件
├── example1.cpp    # 单进程多 GPU
├── example2.cpp    # 多进程单 GPU (MPI)
├── example3.cpp    # 多进程多 GPU (MPI)
└── example4.cpp    # 多通信器 (MPI)
```

## 参考链接

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL API Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [NCCL and MPI](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html)


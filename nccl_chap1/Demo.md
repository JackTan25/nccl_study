# Demo 程序说明

## 什么是 MPI？

**MPI (Message Passing Interface)** 是一种消息传递接口标准，用于编写并行程序。

### 核心概念

| 概念 | 说明 |
|------|------|
| **进程 (Process)** | MPI 程序由多个独立进程组成，每个进程有独立的内存空间 |
| **Rank** | 进程的唯一标识符，从 0 开始编号 |
| **Communicator** | 通信器，定义了哪些进程可以互相通信，`MPI_COMM_WORLD` 是默认的全局通信器 |
| **消息传递** | 进程之间通过发送/接收消息来交换数据 |

### 为什么需要 MPI？

```
┌─────────────────────────────────────────────────────┐
│                    单机多 GPU                        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │
│  │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│                   │
│  └─────┘ └─────┘ └─────┘ └─────┘                   │
│       └───────┬───────┘       └───────┬───────┘    │
│           进程 0 (Rank 0)         进程 1 (Rank 1)   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                    多机多 GPU                        │
│  ┌──────────────────┐    ┌──────────────────┐      │
│  │     节点 A        │    │     节点 B        │      │
│  │ GPU 0  GPU 1     │    │ GPU 0  GPU 1     │      │
│  │ Rank 0  Rank 1   │    │ Rank 2  Rank 3   │      │
│  └──────────────────┘    └──────────────────┘      │
│            │                      │                 │
│            └───── 网络通信 ────────┘                 │
└─────────────────────────────────────────────────────┘
```

**MPI 的作用**：
1. 管理多个进程的启动和生命周期
2. 提供进程间通信的标准接口
3. 支持跨节点（多机）的分布式计算

### MPI 常用函数

| 函数 | 作用 |
|------|------|
| `MPI_Init` | 初始化 MPI 环境 |
| `MPI_Finalize` | 结束 MPI 环境 |
| `MPI_Comm_rank` | 获取当前进程的 Rank |
| `MPI_Comm_size` | 获取总进程数 |
| `MPI_Send / MPI_Recv` | 点对点通信（发送/接收） |
| `MPI_Bcast` | 广播：一个进程发送数据给所有进程 |
| `MPI_Allgather` | 全收集：收集所有进程的数据到所有进程 |
| `MPI_Barrier` | 同步屏障：等待所有进程到达 |

### MPI 与 NCCL 的关系

```
┌────────────────────────────────────────┐
│              应用程序                   │
├────────────────────────────────────────┤
│  MPI：进程管理 + CPU 间通信             │
│  - 启动/管理多个进程                    │
│  - 广播 NCCL UniqueId                  │
│  - 同步进程状态                         │
├────────────────────────────────────────┤
│  NCCL：GPU 间高性能集合通信             │
│  - AllReduce, Broadcast, AllGather...  │
│  - 利用 NVLink/PCIe 高速通信           │
└────────────────────────────────────────┘
```

**分工**：
- **MPI** 负责进程管理和 CPU 层面的协调
- **NCCL** 负责 GPU 之间的高性能数据传输

---

## 运行命令解析

```bash
mpirun -np 2 --allow-run-as-root ./build/demo
```

| 部分 | 含义 |
|------|------|
| `mpirun` | MPI 的启动器，用于启动多个并行进程 |
| `-np 2` | **n**umber of **p**rocesses = 2，启动 2 个 MPI 进程 |
| `--allow-run-as-root` | 允许以 root 用户运行（默认禁止，安全考虑）|
| `./build/demo` | 要运行的程序 |

## 执行流程

```
mpirun -np 2 ./demo
        │
        ▼
┌───────────────────────────────────┐
│  启动 2 个独立的 demo 进程         │
├───────────────────────────────────┤
│  进程 0 (Rank 0)  │  进程 1 (Rank 1) │
│  使用 GPU 0, 1    │  使用 GPU 2, 3   │
└───────────────────────────────────┘
        │
        ▼
   NCCL AllReduce 跨 4 个 GPU 通信
        │
        ▼
  [MPI Rank 0] Success
  [MPI Rank 1] Success
```

## 关键点

1. **-np 2** 表示启动 2 个进程
2. demo.cpp 中 `nDev = 2`，意味着每个进程使用 2 个 GPU
3. 所以总共使用 2 × 2 = **4 个 GPU**

## 调整进程数

如果你想用更多/更少进程，可以调整 `-np` 的值（但需要有足够的 GPU）：

```bash
# 1 个进程，使用 2 个 GPU
mpirun -np 1 --allow-run-as-root ./build/demo

# 2 个进程，使用 4 个 GPU（当前配置）
mpirun -np 2 --allow-run-as-root ./build/demo
```

## 代码核心逻辑

### 1. MPI 初始化

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myRank);   // 获取当前进程编号
MPI_Comm_size(MPI_COMM_WORLD, &nRanks);   // 获取总进程数
```

### 2. 计算 localRank

```cpp
// 通过主机名 hash 计算本地 rank，用于选择 GPU
uint64_t hostHashs[nRanks];
getHostName(hostname, 1024);
hostHashs[myRank] = getHostHash(hostname);
MPI_Allgather(...);  // 收集所有进程的主机 hash
```

### 3. 分配 GPU 资源

```cpp
int nDev = 2;  // 每个进程使用 2 个 GPU
for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(localRank * nDev + i);  // 选择 GPU
    cudaMalloc(sendbuff + i, size * sizeof(float));
    cudaMalloc(recvbuff + i, size * sizeof(float));
}
```

### 4. NCCL 通信器初始化

```cpp
// 生成唯一 ID 并广播给所有进程
if (myRank == 0) ncclGetUniqueId(&id);
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

// 初始化 NCCL 通信器
ncclGroupStart();
for (int i = 0; i < nDev; i++) {
    cudaSetDevice(localRank * nDev + i);
    ncclCommInitRank(comms + i, nRanks * nDev, id, myRank * nDev + i);
}
ncclGroupEnd();
```

### 5. 执行 AllReduce

```cpp
ncclGroupStart();
for (int i = 0; i < nDev; i++) {
    ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]);
}
ncclGroupEnd();
```

**AllReduce 操作**：将所有 GPU 上的数据求和，然后将结果广播到所有 GPU。

### 6. 清理资源

```cpp
// 同步 CUDA 流
for (int i = 0; i < nDev; i++)
    cudaStreamSynchronize(s[i]);

// 释放内存
for (int i = 0; i < nDev; i++) {
    cudaFree(sendbuff[i]);
    cudaFree(recvbuff[i]);
}

// 销毁 NCCL 通信器
for (int i = 0; i < nDev; i++)
    ncclCommDestroy(comms[i]);

// 结束 MPI
MPI_Finalize();
```


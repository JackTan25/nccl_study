/**
 * Example 1: Single Process, Single Thread, Multiple Devices
 *
 * 单进程、单线程管理多个 GPU 的 NCCL 示例
 *
 * 特点:
 *   - 不需要 MPI
 *   - 使用 ncclCommInitAll() 一次性初始化所有通信器
 *   - 使用 Group API 管理多 GPU 操作
 *
 * Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 */
#include <stdlib.h>
#include <stdio.h>
#include "nccl.h"
#include "cuda_runtime.h"


/**
 * CUDA 错误检查宏
 */
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  // ncclComm_t: NCCL 通信器句柄
  // - 标识一组参与通信的 GPU
  // - 每个 GPU 需要一个通信器
  // - 通信器绑定到特定 GPU
  ncclComm_t comms[4];


  // 管理 4 个 GPU 设备
  int nDev = 4;
  int size = 32*1024*1024;  // 每个缓冲区 32M 个 float 元素
  int devs[4] = { 0, 1, 2, 3 };  // GPU 设备 ID 列表


  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }


  // 初始化 NCCL 通信器
  // ncclCommInitAll: 一次性初始化所有通信器（仅适用于单进程多 GPU）
  // 参数:
  //   comms  - 通信器数组，将被初始化
  //   nDev   - GPU 数量
  //   devs   - GPU 设备 ID 数组
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));


  // ============================================================
  // 调用 NCCL 集合通信 API
  // 当单线程管理多个 GPU 时，必须使用 Group API
  // ============================================================

  // ncclGroupStart: 开始一组 NCCL 操作
  // 将多个 NCCL 调用组合在一起，确保它们作为一个整体执行
  NCCLCHECK(ncclGroupStart());

  for (int i = 0; i < nDev; ++i) {
    // ncclAllReduce - 全规约: 所有GPU数据规约后广播到所有GPU
    // 参数: sendbuff(发送缓冲区), recvbuff(接收缓冲区), count(元素数量,非字节), datatype, op, comm, stream
    // datatype可选: ncclInt8/32/64, ncclFloat16/32/64, ncclBfloat16
    // op可选: ncclSum(求和), ncclProd(求积), ncclMax(最大), ncclMin(最小), ncclAvg(平均,NCCL2.10+)
    NCCLCHECK(ncclAllReduce(
        (const void*)sendbuff[i],  // 发送缓冲区: 本GPU要参与规约的数据
        (void*)recvbuff[i],        // 接收缓冲区: 存放规约结果,所有GPU收到相同结果
        size,                       // 元素数量: 32*1024*1024个float (非字节数!)
        ncclFloat,                  // 数据类型: float
        ncclSum,                    // 规约操作: 求和
        comms[i],                   // 通信器: 标识GPU所属通信组
        s[i]));                     // CUDA流: 操作异步加入此流执行
  }

  // ncclGroupEnd: 结束一组 NCCL 操作
  // 此时所有操作被提交到各自的 CUDA 流中异步执行
  NCCLCHECK(ncclGroupEnd());


  // 同步 CUDA 流，等待 NCCL 操作完成
  // 注意: NCCL 操作是异步的，需要同步才能确保完成
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }


  // 释放设备内存
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  // 销毁 NCCL 通信器
  // 注意: 必须在所有使用该通信器的操作完成后才能销毁
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
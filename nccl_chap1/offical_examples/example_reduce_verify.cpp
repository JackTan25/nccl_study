/**
 * 规约操作验证示例
 *
 * 验证 NCCL 各种规约操作的正确性:
 *   - ncclSum  (求和)
 *   - ncclProd (求积)
 *   - ncclMax  (最大值)
 *   - ncclMin  (最小值)
 *   - ncclAvg  (平均值, NCCL 2.10+)
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "nccl.h"
#include "cuda_runtime.h"

#define CUDACHECK(cmd) do { \
  cudaError_t err = cmd; \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t res = cmd; \
  if (res != ncclSuccess) { \
    printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

// 验证结果的容差 (浮点精度)
#define EPSILON 1e-5

// 比较浮点数
bool floatEqual(float a, float b) {
  return fabs(a - b) < EPSILON;
}

// 测试结构体
struct TestResult {
  const char* name;
  bool passed;
  float expected;
  float actual;
};

int main(int argc, char* argv[])
{
  int nDev = 4;  // 使用 4 个 GPU
  int count = 8; // 每个缓冲区 8 个元素 (便于验证)

  printf("=== NCCL 规约操作验证 ===\n");
  printf("GPU 数量: %d, 元素数量: %d\n\n", nDev, count);

  ncclComm_t comms[4];
  int devs[4] = {0, 1, 2, 3};

  // 分配缓冲区
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  float** hostSend = (float**)malloc(nDev * sizeof(float*));  // CPU 端数据
  float** hostRecv = (float**)malloc(nDev * sizeof(float*));  // CPU 端结果
  cudaStream_t* streams = (cudaStream_t*)malloc(nDev * sizeof(cudaStream_t));

  // 初始化每个 GPU 的数据
  // GPU 0: [1, 2, 3, 4, 5, 6, 7, 8]
  // GPU 1: [2, 3, 4, 5, 6, 7, 8, 9]
  // GPU 2: [3, 4, 5, 6, 7, 8, 9, 10]
  // GPU 3: [4, 5, 6, 7, 8, 9, 10, 11]
  for (int i = 0; i < nDev; ++i) {
    hostSend[i] = (float*)malloc(count * sizeof(float));
    hostRecv[i] = (float*)malloc(count * sizeof(float));
    for (int j = 0; j < count; ++j) {
      hostSend[i][j] = (float)(i + j + 1);
    }

    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&sendbuff[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff[i], count * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff[i], hostSend[i], count * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // 打印初始数据
  printf("初始数据:\n");
  for (int i = 0; i < nDev; ++i) {
    printf("  GPU %d: [", i);
    for (int j = 0; j < count; ++j) {
      printf("%.0f%s", hostSend[i][j], j < count-1 ? ", " : "");
    }
    printf("]\n");
  }
  printf("\n");

  // 初始化 NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // ============================================================
  // 测试 1: ncclSum (求和)
  // ============================================================
  printf("--- 测试 ncclSum (求和) ---\n");
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count, ncclFloat, ncclSum, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  // 同步并拷回结果
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    CUDACHECK(cudaMemcpy(hostRecv[i], recvbuff[i], count * sizeof(float), cudaMemcpyDeviceToHost));
  }

  // 验证 Sum: 预期结果 = GPU0[j] + GPU1[j] + GPU2[j] + GPU3[j]
  printf("  预期: [");
  for (int j = 0; j < count; ++j) {
    float expected = 0;
    for (int i = 0; i < nDev; ++i) expected += hostSend[i][j];
    printf("%.0f%s", expected, j < count-1 ? ", " : "");
  }
  printf("]\n");

  printf("  实际: [");
  for (int j = 0; j < count; ++j) {
    printf("%.0f%s", hostRecv[0][j], j < count-1 ? ", " : "");
  }
  printf("]\n");

  bool sumPassed = true;
  for (int j = 0; j < count; ++j) {
    float expected = 0;
    for (int i = 0; i < nDev; ++i) expected += hostSend[i][j];
    if (!floatEqual(hostRecv[0][j], expected)) sumPassed = false;
  }
  printf("  结果: %s\n\n", sumPassed ? "✓ 通过" : "✗ 失败");

  // ============================================================
  // 测试 2: ncclProd (求积)
  // ============================================================
  printf("--- 测试 ncclProd (求积) ---\n");

  // 重新拷贝数据
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(sendbuff[i], hostSend[i], count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count, ncclFloat, ncclProd, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    CUDACHECK(cudaMemcpy(hostRecv[i], recvbuff[i], count * sizeof(float), cudaMemcpyDeviceToHost));
  }

  // 验证 Prod
  printf("  预期: [");
  for (int j = 0; j < count; ++j) {
    float expected = 1;
    for (int i = 0; i < nDev; ++i) expected *= hostSend[i][j];
    printf("%.0f%s", expected, j < count-1 ? ", " : "");
  }
  printf("]\n");

  printf("  实际: [");
  for (int j = 0; j < count; ++j) {
    printf("%.0f%s", hostRecv[0][j], j < count-1 ? ", " : "");
  }
  printf("]\n");

  bool prodPassed = true;
  for (int j = 0; j < count; ++j) {
    float expected = 1;
    for (int i = 0; i < nDev; ++i) expected *= hostSend[i][j];
    if (!floatEqual(hostRecv[0][j], expected)) prodPassed = false;
  }
  printf("  结果: %s\n\n", prodPassed ? "✓ 通过" : "✗ 失败");

  // ============================================================
  // 测试 3: ncclMax (最大值)
  // ============================================================
  printf("--- 测试 ncclMax (最大值) ---\n");

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(sendbuff[i], hostSend[i], count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count, ncclFloat, ncclMax, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    CUDACHECK(cudaMemcpy(hostRecv[i], recvbuff[i], count * sizeof(float), cudaMemcpyDeviceToHost));
  }

  printf("  预期: [");
  for (int j = 0; j < count; ++j) {
    float expected = hostSend[0][j];
    for (int i = 1; i < nDev; ++i) if (hostSend[i][j] > expected) expected = hostSend[i][j];
    printf("%.0f%s", expected, j < count-1 ? ", " : "");
  }
  printf("]\n");

  printf("  实际: [");
  for (int j = 0; j < count; ++j) {
    printf("%.0f%s", hostRecv[0][j], j < count-1 ? ", " : "");
  }
  printf("]\n");

  bool maxPassed = true;
  for (int j = 0; j < count; ++j) {
    float expected = hostSend[0][j];
    for (int i = 1; i < nDev; ++i) if (hostSend[i][j] > expected) expected = hostSend[i][j];
    if (!floatEqual(hostRecv[0][j], expected)) maxPassed = false;
  }
  printf("  结果: %s\n\n", maxPassed ? "✓ 通过" : "✗ 失败");

  // ============================================================
  // 测试 4: ncclMin (最小值)
  // ============================================================
  printf("--- 测试 ncclMin (最小值) ---\n");

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(sendbuff[i], hostSend[i], count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count, ncclFloat, ncclMin, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    CUDACHECK(cudaMemcpy(hostRecv[i], recvbuff[i], count * sizeof(float), cudaMemcpyDeviceToHost));
  }

  printf("  预期: [");
  for (int j = 0; j < count; ++j) {
    float expected = hostSend[0][j];
    for (int i = 1; i < nDev; ++i) if (hostSend[i][j] < expected) expected = hostSend[i][j];
    printf("%.0f%s", expected, j < count-1 ? ", " : "");
  }
  printf("]\n");

  printf("  实际: [");
  for (int j = 0; j < count; ++j) {
    printf("%.0f%s", hostRecv[0][j], j < count-1 ? ", " : "");
  }
  printf("]\n");

  bool minPassed = true;
  for (int j = 0; j < count; ++j) {
    float expected = hostSend[0][j];
    for (int i = 1; i < nDev; ++i) if (hostSend[i][j] < expected) expected = hostSend[i][j];
    if (!floatEqual(hostRecv[0][j], expected)) minPassed = false;
  }
  printf("  结果: %s\n\n", minPassed ? "✓ 通过" : "✗ 失败");

  // ============================================================
  // 测试 5: ncclAvg (平均值, NCCL 2.10+)
  // ============================================================
  printf("--- 测试 ncclAvg (平均值) ---\n");

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(sendbuff[i], hostSend[i], count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count, ncclFloat, ncclAvg, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    CUDACHECK(cudaMemcpy(hostRecv[i], recvbuff[i], count * sizeof(float), cudaMemcpyDeviceToHost));
  }

  printf("  预期: [");
  for (int j = 0; j < count; ++j) {
    float expected = 0;
    for (int i = 0; i < nDev; ++i) expected += hostSend[i][j];
    expected /= nDev;
    printf("%.2f%s", expected, j < count-1 ? ", " : "");
  }
  printf("]\n");

  printf("  实际: [");
  for (int j = 0; j < count; ++j) {
    printf("%.2f%s", hostRecv[0][j], j < count-1 ? ", " : "");
  }
  printf("]\n");

  bool avgPassed = true;
  for (int j = 0; j < count; ++j) {
    float expected = 0;
    for (int i = 0; i < nDev; ++i) expected += hostSend[i][j];
    expected /= nDev;
    if (!floatEqual(hostRecv[0][j], expected)) avgPassed = false;
  }
  printf("  结果: %s\n\n", avgPassed ? "✓ 通过" : "✗ 失败");

  // ============================================================
  // 总结
  // ============================================================
  printf("=== 测试总结 ===\n");
  printf("  ncclSum:  %s\n", sumPassed ? "✓ 通过" : "✗ 失败");
  printf("  ncclProd: %s\n", prodPassed ? "✓ 通过" : "✗ 失败");
  printf("  ncclMax:  %s\n", maxPassed ? "✓ 通过" : "✗ 失败");
  printf("  ncclMin:  %s\n", minPassed ? "✓ 通过" : "✗ 失败");
  printf("  ncclAvg:  %s\n", avgPassed ? "✓ 通过" : "✗ 失败");

  bool allPassed = sumPassed && prodPassed && maxPassed && minPassed && avgPassed;
  printf("\n总体结果: %s\n", allPassed ? "✓ 全部通过" : "✗ 存在失败");

  // 清理
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
    ncclCommDestroy(comms[i]);
    free(hostSend[i]);
    free(hostRecv[i]);
  }
  free(sendbuff);
  free(recvbuff);
  free(hostSend);
  free(hostRecv);
  free(streams);

  return allPassed ? 0 : 1;
}


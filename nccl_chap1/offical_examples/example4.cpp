/**
 * Example 4: Multiple communicators per device
 *
 * This example shows how to create multiple communicators per device,
 * demonstrating both blocking and non-blocking communicator initialization.
 *
 * Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


static uint64_t getHostHash(const char* hostname) {
  uint64_t result = 5381;
  for (int c = 0; hostname[c] != '\0'; c++){
    result = ((result << 5) + result) ^ hostname[c];
  }
  return result;
}


// User-defined timeout check function
static int timeoutCounter = 0;
static bool checkTimeout() {
  timeoutCounter++;
  // Timeout after 60000 iterations (about 60 seconds with 1ms sleep)
  return timeoutCounter > 60000;
}


int main(int argc, char* argv[])
{
    int myRank, nRanks, localRank = 0;

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // Calculate localRank for GPU selection
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
       if (p == myRank) break;
       if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    // Number of communicators per device
    const int commNum = 2;
    ncclUniqueId id;
    ncclComm_t blockingComms[commNum];
    ncclComm_t nonblockingComms[commNum];
    ncclResult_t state;

    // Set device
    CUDACHECK(cudaSetDevice(localRank));

    // Create blocking communicators
    printf("[Rank %d] Creating %d blocking communicators...\n", myRank, commNum);
    for (int i = 0; i < commNum; ++i) {
        if (myRank == 0) ncclGetUniqueId(&id);
        MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
        NCCLCHECK(ncclCommInitRank(&blockingComms[i], nRanks, id, myRank));
    }
    printf("[Rank %d] Blocking communicators created.\n", myRank);

    // Create non-blocking communicators
    printf("[Rank %d] Creating %d non-blocking communicators...\n", myRank, commNum);
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    for (int i = 0; i < commNum; ++i) {
        if (myRank == 0) ncclGetUniqueId(&id);
        MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

        // Non-blocking init returns ncclInProgress immediately, which is expected
        ncclResult_t initResult = ncclCommInitRankConfig(&nonblockingComms[i], nRanks, id, myRank, &config);
        if (initResult != ncclSuccess && initResult != ncclInProgress) {
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(initResult));
            exit(EXIT_FAILURE);
        }

        // Wait for non-blocking init to complete
        timeoutCounter = 0;
        state = ncclInProgress;
        do {
            ncclCommGetAsyncError(nonblockingComms[i], &state);
            if (state != ncclSuccess && state != ncclInProgress) {
                printf("[Rank %d] Non-blocking comm %d error: %s\n",
                       myRank, i, ncclGetErrorString(state));
                break;
            }
            usleep(1000);  // Sleep 1ms to avoid busy waiting
        } while(state == ncclInProgress && checkTimeout() != true);

        if (state == ncclInProgress) {
            printf("[Rank %d] Non-blocking comm %d timed out, still in progress\n", myRank, i);
        }
    }
    printf("[Rank %d] Non-blocking communicators created.\n", myRank);

    // Destroy all communicators
    for (int i = 0; i < commNum; ++i) {
        ncclCommDestroy(blockingComms[i]);
        ncclCommDestroy(nonblockingComms[i]);
    }

    // Finalize MPI
    MPICHECK(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", myRank);
    return 0;
}

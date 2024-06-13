# How Sgemm is Optimized 
![Diagram of `C = AB` ](https://private-user-images.githubusercontent.com/61581888/339216277-b5190968-244d-4215-9812-f22734ef166e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTgyNTQ2OTYsIm5iZiI6MTcxODI1NDM5NiwicGF0aCI6Ii82MTU4MTg4OC8zMzkyMTYyNzctYjUxOTA5NjgtMjQ0ZC00MjE1LTk4MTItZjIyNzM0ZWYxNjZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjEzVDA0NTMxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjMzlhMjgxYmRmNmUxNWE5YzA5ZmE0ZmEyYWUyMWQzNjAwODQ4ZDRkM2ZiNWIyNmY3ZTIyMzBjZmFhYWQ3MTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.01zt5DtxhA8opzdce5u7igBEcM9CTKxV5nKxngyibhE)

## Optimized Codes in `sgemm.cu`
We already know that 
* `A` is a matrix with dim (M * K).
* `B` is a matrix with dim (K * N).
* Result `C` is a matrix with dim (M * K)

I have to say that one quesiton that confused me for a long time is:
Even though CUDA defines the dimension in 3D which not means the CUDA's correpoinding blocks or GPU cores are organized in cube. It defines as 3D just want to define different dimensions of {x, y, z} as different unit of parallelism easy to distinguish(and also easy to get confused for beginner TAT/~ ). 

---- 

## Different Hierarchy Compute Components in CUDA
* Hirarchy of compute components defined in CUDA 
```
Grid > Block > Thread
```
> Grid: The highest level of organizaiton. A grid contians multiple blocks. 
> Block: A grid is divided into blocks. Each block contains multiple threads. 
> Thread: The smallest unit of computation within a block. 

* Fine-Grained Management Concepts in CUDA 
To provide more fine-grined management import some logical concepts
Grid > Group > Block > Warp(tile) > Thread 

> Grid: A collection of blocks that execute a kernel.
> Block: A collection of threads that execute on a single multiprocessor(SM) and can share data through shared memory.
> Warp: A group of 32 threads within a block that execute instructions in lockstep.
> Thread: The smallest unit of execution.

The term "Group" in `group cooperative` isn't typically used in CUDA documentation. 
Instead, focus on understanding the relationship between grids, blocks, warps, and threads. 
Warps are a critical concept for understanding performance, as threads within a warp execute synchronously. 

## Cooperative Behavior in CUDA 
CUDA provides mechanisms for threads to cooperate and synchronize at different granularities:
### Block-Level Cooperation 
* Threads within a block can share data via shared memory
* Threads within a block can synchornize at specific points using `__syncthreads()`. This function to ensure that all threads in the block reach a synchronization point before any thread can proceed. 

### Warp-Level Cooperation 
* Threads within a warp executein lockstep and can use warp-level primitives for efficient communication and synchronizaiton.
* Functins like `__shfl_sync()` , `__ballot_sync()`, and others enable cooperation among threads within a warp without the overhead of block-wide synchronization.

---- 

## Different Hierarchy Compute Components' Synchronization Methods 
CUDA provides several layers of parallelism, and each layer has its own synchronization methods to ensure threads of blocks cna coordinate their execution.
Below, outlines the different components, their corresponding synchronization methods, and comon scenarios where these synchornizations are used. 

### Grid-Level Sync 
* Description:
A grid is a collection of blocks. Each block in the grid operates independently. 
There is no direct synchronization mechanism provided by CUDA for synchronization between blocks within a grid. 

* Common Scenario: 
Synchornization across the grid can be achieved by breaking the kernel into multiple kernel launches. For instance, if some data needs to be processed by all blocks before the next stage of processing, you have to end the current kernel and start a new one. 

### Block-Level Sync 
* Description:
A block a collection of threads. Threads within a block can share data through shared memory and can synchronize their execution. 

* Kernel provides naive Method 
```
__syncthreads(); 
```

* Common Scenario:
Codes below is used in scenarios where threads within a block need to collaborate.
For example, in matrix multiplication using shared memory, you might load tiles of matrix
into shared memory, synchronize all threads in the block to ensure loading is complete, 
and then proceed with the computation.
```cudas
__global__
void matrixMulKernel(float *C, float *A, float *B, int N) 
{
    __shared__ float _S_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float _S_B[BLOCK_SIZE][BLOCK_SIZE]; 

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int row = blockIdx.y * BLOCK_SIZE + ty; 
    int col = blockIdx.x * Block_SIZE + tx; 

    float res = 0.f; 
    for (int idx = 0; idx < N / BLOCK_SIZE; idx++) {
        _S_A[ty][tx] = A[row * N + idx * BLOCK_SIZE + tx]; 
        _S_B[ty][tx] = B[(idx * BLOCK_SIZE + ty) * N + col];

        // here wait all threads write A, B corresponding data value 
        // --> shared memory all get ready 
        __syncthreads();

        // threads with different threadIdx.x threadIdx.y 
        // can access different range of the matrix
        // then accumulate values by different threads
        // accumulate data from different regions of the shared memory to the final result 

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            res += _S_A[ty][k] * _S_B[k][tx]; 
        }

        // here call sync to make sure that all values from shared memory 
        // are retrieved and accumulate to final result res 
        __syncthreads(); 
    }

    // finally write the correspoinding row, col of A and B (that executed in parallel)
    // to one C matrix's value 
    C[row * N + col] = res; 
}
```

* Common Scenario:


### Warp-Level Sync 
### Thread-Level Sync 

## Common Scenarios for CUDA Synchronization 


---- 
## Understand Different Types of Index in CUDA 
### `GridIdx`
### `BlockIdx`
* `blockIdx`: This is CUDA inner defined 3 dim variable, which includes current thread block(Block)'s index in the scope of Grid. 

* `blockIdx.x`: Current Block's index in the X axis of Grid. 
* `blockIdx.y`: Current Block's index in the X axis of Grid. 

### `BlockDim`
* `blockDim`: This is a CUDA inner defined 3 dim variable, which determines how many threads per block. 
* `blockDim.x`: How many threads available on the X axis in current Block.
* `blockDim.y`: How many threads available on the Y axis in current Block. 

### `ThreadIdx`
* `threadIdx`: CUDA inner defined 3 dim variable, which present current thread's index in the scope of Block.
* `threadIdx.x`: current active thread's index in the X axis of current Block.
* `threadIdx.y`: current active thread's index in the Y axis of current Block. 


* `BLOCK_DIM`: dimension of the thread block which means block thread dimension is 16, each block has 16 * 16 = 256 in total threads. 

* `blockIdx.x`: means one block's thread index in the dimension of X axis. Since we set the `BLOCK_DIM` as 16, so we know that in the scope of block `blockIdx.x` value should be in the range of `[0, 16 - 1]`.

* `blockIdx.y`: means one block's thread index in the dimension of Y axis. Since we set the `BLOCK_DIM` as 16, so we know that in the scope of block `blockIdx.y` value should be in the range of `[0, 16 - 1]`.

## Understand Different Types of Index in CUDA 


```cuda
#define BLOCK_DIM 16   

__global__
void sgemm_kernel_opt_version(const float *a, const float *B, float *C,
            int M, int N, int K, float alpha, float beta)
{
    int bid_x = blockIdx.x * blockDim.x; 
    int bid_y = blockIdx.y * blockDim.y; 
    int tid_x = threadIdx.x; 
    int tid_y = threadIdx.y; 

    float element_c = 0.f; 
    __shared__ float s_title_A[BLOCK_DIM][BLOCK_DIM]; 
    __shared__ float s_title_B[BLOCK_DIM][BLOCK_DIM]; 

    // forward tile with size in matrix A
    for (int k = 0; k < K)

}            
```
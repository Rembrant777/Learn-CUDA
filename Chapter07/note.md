# Chapter07 Parallel Programming Patterns in CUDA 

## Why CUDA defines Multi-Dimension to Support (Parallel) Computing?
CUDA adopts Multi-Deimension(1D, 2D and 3D) with the aim to support 
different data structures. 

(其实这里最终的目的就是在计算模型这里提供不同 dim 的并行粒度的可能, 
比如我有一个 cube 这种数据结构, 然后有了  cube 这种支持 3D 的计算模型基础
就是申请 1 个 block, 那么 
`blockId.x` 用来作为 X dim 的所有计算操作 - 并行粒度 1
`blockId.y` 用来作为 Y dim 的所有计算操作 -  并行粒度 2 
`blockId.z` 用来作为 Z dim 的所有计算操作 - 并行粒度 3
)

这么搞的话, 3 个并行粒度, 除了中间计算结果通过 group cooperation 还是 block scoped thread sync 进行同步即可. 

至于更高的计算维度, 完全可以通过求导将高阶的计算拍平到 3D 再发起计算, 3D 的并发粒度求不同的数学公式完全是足够用的.

对的对的, 如果是高维 4 维的数组的话完全可以将其转换为 3 维 的再通过 CUDA 来进行计算. 
```
A[w][x][y][z] -> B[w*x][y][z]
```

#### 1D
* 1 dimension data structure, like array or vector.
* to solve such kind of data structure's computation, we can use 1D Block's threads to solve. 

#### 2D
* 2 dimension data structure, like matrix or image(arrays)
* to solve such kind of data 

#### 3D
* 3D images, cube data or physical simulation

---

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

---

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

### Warp-Level Sync 
* Description: 
A warp consists of 32 threads within a block that execute instructions in lockstep.
CUDA provides warp-level intrinsics to enable fine-grained communication and synchronization within a warp. 

* Method:
`__shfl_sync()`, `__ballot_sync()`

* Common Scenario:
Warp-level primitives are used for more efficient and fine-grained synchronization than block-wide synchronzation. 
For example, in algorithms like reduction, prefix sum, and certain machine learning algorithms, these instrinsics are used to communicate and synchronzie between threads within a warp. 
```cuda
__global__ void warpReduction(float *input, float *output) {
    int tid = threadIdx.x;
    float val = input[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid % warpSize == 0) output[tid / warpSize] = val;
}
```

### Thread-Level Sync 
* Description:
The basic unit of execution. While threads do not synchronize on their own,
they can be coordinate using atomic operations. 

* Methods:
`atomicAdd()`, `atomicSub()`

* Common Scenario:
Atomic operations are often used in scenarios where multiple threads need to 
safely update shared data without race conditions. For example, counting occurences 
of certain values in a large dataset or safely accumulating results from multiple threads. 
```cuda 
__global__
void countOccurences(int *data, int *count, int value)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (data[idx] = =value) {
        atomicAdd(count, 1); 
    }
}
```

--- 

## Common Scenarios for CUDA Synchronization 
* Matrix Multiplication: 
Synchronization within blocks to load tiles of matrices into shared memory and ensure all threads are ready before proceeding with computation.

* Reduction: 
Using warp-level primitives for efficient parallel reduction operations.

* Histogram Calculation: 
Using atomic operations to safely update bin counts from multiple threads.

* Prefix Sum (Scan): 
Using warp and block-level synchronization to compute prefix sums efficiently.

* Convolution Operations: 
Synchronizing threads within a block to load input data into shared memory before applying convolution filters.

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

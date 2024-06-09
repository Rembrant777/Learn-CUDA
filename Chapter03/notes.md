# Chapter03 CUDA Thread Programming 

### Warp in CUDA Architecture 
```
Warp is a significant concept in GPU architecture. Warp is a concept from the logical layer, but also referred in physical layer. 

From the logical layer, warp is referring to a scheduler unit in GPU. 
In the architecture of NVIDIA CUDA, a warp usually contains 32 threads. 

Those threads are scheduled and executed together as a unit. All threads in a warp execute the same instructions, but they handles different data.(SIMT single instruction multiple threads).

From the physical layer, warp shows its hardware layer’s execution strategy. GPU’s compute unit(also know as the streaming multi-processor or CUDA core) is designed to execute warp’s inner threads. One warp’s all threads on the hardware layer are execute in parallel. GPU hardware has its specified circuit components to handle warp grained schedule, execute and memory access. 
```

### Parallel and Concurrent in CUDA 
```
What's interesting in CUDA execution is:
> CUDA threads exeuctes in parallel.
> CUDA blocks operate in concurrent. 
```

### Understand Grid and Warp in CUDA
```
CUDA Grid   -> Spark Job
CUDA Block  -> Spark Stage 
CUDA Warp   -> Spark Executor's Thread Pool
CUDA Thread -> Spark Task // minimum executable unit defined in Spark Schedule Architecture 
```

* CUDA Grid : Spark Job
```
Grid in GPU represents a big compute job in which contains multiple Blocks, just like a Spark Job. 
```
* CUDA Block : Spark Stage 
```
Block in GPU's Grid just like Stage defined in Spark Job.

One CUDA Block contains multiple Thread(s), just like one Spark Stage contains multiple Task(s).

Threads in each CUDA Block can cooperate with each other, and share shared memory. 
Similariy, Spark Stage's Task(s) can share data via shuffle operation(require network communication depends on specific compute strategy).
```

* CUDA Warp : Spark Executor(thread pool)
```
CUDA Warp is hardware component, usually contains 32 threads(depend on GPU type). 
Just like the thread pool in Spark Executor. 
```

* CUDA Thread : Spark Task 
```
CUDA Thread is the minimum grain component to execute and handle specific compute task. 
Just like the Spark Task is also the minimum grain component to execute and handle partition data. 
But, CUDA Thread is the resource concept, and Spark Task is the schedule concept. 

Spark Task need to be allocated thread from the Spark Executor then it can be executed. 

Spark Executor's thread pool also decides the parallelism of the Spark Stage. 

And there is also a big difference between the CUDA Thread and Spark Task. 

Spark Task's executor the thread is apply from the Executor's JVM thread pool, and also be execute in the scope of JVM. 

And the CUDA Thread is apply from the GPU which is the hardware layer's thread. 
```

### Understand CUDA Thread's `threadIdx.x` and CUDA Lane `LaneIndex` in CUDA 
* Thread Index
```
threadIdx.x means thread index of the block.
we can regard it as the global thread index from the block, cuz one block can contain one or more warps.
```


* Lane 
```
lane index = threadIdx.x & (warpSize -1) menas current thread's correspoinding local index in the scope of warp.
we can regard it as the local thread inde from the grain of warp, cuz one block can contains one or more warps. 
```

* Difference or Relationship between Thread Idx and Lane Idx
```
Thread Idx is in the grain of Block, it is the index id of total thread counts allocated to the block.

Lane Idx is in the scope of Warp, it is the index id of total thread counts allocated to the warp. 
```

### CUDA occupancy
```
CUDA occupancy is the ratio of active CUDA warps to the maximum warps that each streaming multi-processor can execute concurrently.

Higher occupancy means higher effective GPU utilization. 

Developers can determine CUDA occupancy using two methods:
1. Theoretical occcupancy
> Theoretical occupancy determined by the CUDA Occupancy Calculator: An Excel sheet provided by the CUDA Toolkit. 
> Theoretical occupancy can be determined from each kernel's resource usage and GPU's streaming multiprocessor. 
> Theoretical occupancy can be regarded as the maximum upper-bound occupancy because the occupancy number does not consider instructional dependencies or memory bandwidth limitations. 


2. Achieved occupancy
Achieved Occupancy reflects the true number of concurrent executed warps on a streaming multiprocessor and the maximum available warps. 

Achieved occupancy can be measured by the NVIDIA profiler with metric analysis. 
```

### Different Indexes and Concepts in CUDA Programming
* Block Index 
* Warp Index 
* Lane Index
* Thread Index
* BlockDim 


### Enable NVCC report GPU resource usage
* 
```shell
nvcc -m 64 --resource-usage \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -I/usr/local/cuda/samples/common/inc \
    -o sgemm ./sgemm.cu 
```

* NVCC GPU resource usage doc: [link](https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html#building-turing-compatible-apps-using-cuda-10-0)

### Parallel Reduction 
```
Reduction is a simple but useful algorithm to obtain a common parameter across many parameters. 

Reduction tasks can be executed in sequence or in parallel. 

Parallel reduction is the fastest way of getting a histogram, mean, or any other statistical values. 
```

### Shared Memory vs. Global Memory

#### Shared Memory 
* Shared Memory is block grained. All block scoped thread can get access to Shared Memory.
* Shared Memory's bandwidth: TB/s and locates inside of Streaming Multiprocessor. 
* Access Latency: Low

#### Global Memory
* Global Memory can be get access by all the blocks' threads. 
* Global Memory's bandwidth: GB/s to 1TB/s depends on type of GPU.
* Access Latency: High

* Tips

1. Use shared memory as much as possible. When use global memory prefer select coalesced access to maximum utilization of the memory's bandwidth. 
2. constant memory and texture memory can be used to store `only read`` or `repeat read` data. 

```
  +---------------------+       +---------------------+
  |      Block 0        |       |      Block 1        |
  | +-----------------+ |       | +-----------------+ |
  | |   Shared Mem    | |       | |   Shared Mem    | |
  | |  (Block scope)  | |       | |  (Block scope)  | |
  | +-----------------+ |       | +-----------------+ |
  |     +------+         |       |     +------+         |
  |     |Warp 0|         |       |     |Warp 0|         |
  |     +------+         |       |     +------+         |
  |     |Warp 1|         |       |     |Warp 1|         |
  |     +------+         |       |     +------+         |
  +---------------------+       +---------------------+
              |                         |
              |                         |
              +-----------+-------------+
                          |
                    Global Memory
                   (All Blocks scope)

```

### Generate Performance Report Command 
```shell
/usr/local/cuda-9.2/bin/nvprof -o reduction_global.nvvp ./reduction_global
/usr/local/cuda-9.2/bin/nvprof --analysis-metrics -o reduction_global_metric.nvvp ./reduction_global

/usr/local/cuda-9.2/bin/nvprof -o reduction_shared.nvvp ./reduction_shared
/usr/local/cuda-9.2/bin/nvprof --analysis-metrics -o reduction_shared_metric.nvvp ./reduction_shared
```

### Branch Divergence
```
In a single instruction, multiple thread(SIMT) execution model, threads are grouped into set of 32 threads 
and each group is called a wrap. 

If a warp encounters a conditional statement or branch, its threads can be diverged and serialized to 
execute each condition. 

This is called branch divergence, which impacts performance significantly.
```

### Why we need to avoid or minimize warp divergence? 
```
As more of the branched part becomes significant, the GPU scheduling throughput becomes inefficient.
Therefore, we need to avoid or minimize warp divergence effect. 
```

### How to avoid warp divergence?
```
* Divergence avoidance by handling different warps to execute the branched part. 

* Coalsecing the branched part to reduce branches in a warp

* Shortening the branched part; only critical parts to be branched.

* Rearranging the data(transposing, coalescing, and so on)

* Partitioning the group using tiled_partition in Cooperative Group
```





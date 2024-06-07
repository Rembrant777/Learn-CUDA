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
* Grid 
```
```

* Warp 
```
```

* Difference or Relationship between Grid and Warp 
```
```

* Trying to understand Grid and Wrap with the help of Spark Architecutre 
```
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

### Different Indexes and Concepts in CUDA Programming
* Block Index 
* Warp Index 
* Lane Index
* Thread Index
* BlockDim 
* 
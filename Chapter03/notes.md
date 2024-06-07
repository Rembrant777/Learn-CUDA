# Chapter03 CUDA Thread Programming 

* Warp in CUDA Architecture 
```
Warp is a significant concept in GPU architecture. Warp is a concept from the logical layer, but also referred in physical layer. 

From the logical layer, warp is referring to a scheduler unit in GPU. 
In the architecture of NVIDIA CUDA, a warp usually contains 32 threads. 

Those threads are scheduled and executed together as a unit. All threads in a warp execute the same instructions, but they handles different data.(SIMT single instruction multiple threads).

From the physical layer, warp shows its hardware layer’s execution strategy. GPU’s compute unit(also know as the streaming multi-processor or CUDA core) is designed to execute warp’s inner threads. One warp’s all threads on the hardware layer are execute in parallel. GPU hardware has its specified circuit components to handle warp grained schedule, execute and memory access. 
```
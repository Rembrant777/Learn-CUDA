# Chapter 04 Kernel Execution Notes 

## Difference between `malloc` and `cudaMallocHost`
Both `malloc` and `cudaMallocHost` functions allocate memory on host(CPU) side.
But they serve different purpose and have distinct characteristics, especially in the context of CUDA programming. 

### `malloc`
* Purpose:
Allocates memory on the host(CPU) using standard C library functions. 

* Usage:
General-purpose memory allocation for any C/C++ applicaiton. 

* Memory Location:
The allocated memory resides in the host's main memory(RAM).


* Performance:
No special optimizaitons for CUDA. The allocated memory may not be 
contiguous or aligend for optimal GPU access. 

### `cudaMallocHost`
* Purpose :
Allocates pinned(page-locked) memory on the host. 

* Usage :
Specifically for CUDA applications to facilitate and more efficient data transfer
between the host and the device(GPU).

* Memory Location:
The allocated memory resides in the host's main memory (RAM), but it is pinned(page-locked).

* Performance 
> Pinned Memory: The memory can not be paged out by the operating system, which means the GPU can directly access it without involving the CPU.
> Faster Transfers: Data transfers between the host and the device using pinned memory are typically faster compared to pageable memory allocated by `malloc`.
> Asynchronous Transfers: Pinned memory allows for asynchronous memory transfers (using strams), enabling overlapping of computation and data transfer for better performance. 

### Key Differences 
* Memory Type
* Performance Impact
* Usage Context 
* `malloc` is suitable for general-purpose memory allocation in any C/C++ program
* `cudaMallocHost` is designed for CUDA applications where efficient host-device data transfer is critical
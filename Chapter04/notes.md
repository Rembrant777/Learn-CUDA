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

## CUDA Event 
### CUDA Event Usage Scenarioes 
1. Measure Kernel Execute Time in More Fine-Grain
```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;

// calculate time consume and set value -> millisecons 
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %f ms\n", milliseconds);
```
2. Measure Data Transfer 
```cuda
cudaEventRecord(start);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaEventElapsedTime(&milliseconds, start, stop);
printf("Data transfer time: %f ms\n", milliseconds);
```

3. Multi-Stream Synchoronize Signal 
```cuda
// create event 
cudaEvent_t event;
cudaEventCreate(&event);

// copy data in async mode
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);

// record event in the grain of stream1
cudaEventRecord(event, stream1);

// wait event that recorded in the stream1 get ready
// only event status set ok will stream2 begin to execute the my_kernel logic 
cudaStreamWaitEvent(stream2, event, 0);

my_kernel<<<grid, block, 0, stream2>>>(...);
```

### CUDA Event Advantages vs. CPU StopWatchInterface 
1. Higher Precision
2. GPU Operate Orders Oriented
3. Multi Device(Block) and Stream Sync 
4. Non-CPU Interruption 

## CUDA Callback Function Signatre 
```cuda
cudaError_t cudaStreamAddCallback(
    cudaStream_t stream,
    cudaStreamCallback_t callback,
    void *userData,
    unsigned int flags
);
```

Here the param of `cudaStreamCallback_t` is defined as follow:
```cuda 
typedef void (CUDART_CB *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);
```

Actually `cudaStreamCallback_t` is the function pointer which mean parameter to be passed to the 
cudaStreamAddCallback should match the function signature like 
```cuda
// function 
void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *userData);  
```

## Understand CUDA's Dynamic Parallelism 

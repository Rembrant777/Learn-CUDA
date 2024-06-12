# Chapter 04 Kernel Execution Notes 

## Difference between `malloc` and `cudaMallocHost`
Both `malloc` and `cudaMallocHost` functions allocate memory on the host(CPU) side.
But they serve different purpose and have distinct characteristics, especially in the context of CUDA programming. 

### `malloc`
* Purpose:
Allocates memory on the host(CPU) using standard C library functions. 

* Usage:
General-purpose memory allocation for any C/C++ application. 

* Memory Location:
The allocated memory resides in the host's main memory(RAM).


* Performance:
No special optimizations for CUDA. The allocated memory may not be 
contiguous or aligned for optimal GPU access. 

### `cudaMallocHost`
* Purpose :
Allocates pinned(page-locked) memory on the host. 

* Usage :
Specifically for CUDA applications to facilitate more efficient data transfer
between the host and the device(GPU).

* Memory Location:
The allocated memory resides in the host's main memory (RAM), but it is pinned(page-locked).

* Performance 
> Pinned Memory: The memory can not be paged out by the operating system, which means the GPU can directly access it without involving the CPU.
> Faster Transfers: Data transfers between the host and the device using pinned memory are typically faster compared to pageable memory allocated by `malloc`.
> Asynchronous Transfers: Pinned memory allows for asynchronous memory transfers (using streams), enabling overlapping of computation and data transfer for better performance. 

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

//Wait event that was recorded in the stream1 Get ready
// only event status set ok will stream2 begin to execute the my_kernel logic 
cudaStreamWaitEvent(stream2, event, 0);

my_kernel<<<grid, block, 0, stream2>>>(...);
```

### CUDA Event Advantages vs. CPU StopWatchInterface 
1. Higher Precision
2. GPU Operate Orders Oriented
3. Multi Device(Block) and Stream Sync 
4. Non-CPU Interruption 

## CUDA Callback Function Signature 
```cuda
cudaError_t cudaStreamAddCallback(
    cudaStream_t stream,
    cudaStreamCallback_t callback,
    void *userData,
    unsigned int flags
);
```

Here the param of `cudaStreamCallback_t` is defined as follows:
```cuda 
typedef void (CUDART_CB *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void *userData);
```

Actually `cudaStreamCallback_t` is the function pointer which means parameter to be passed to the 
cudaStreamAddCallback should match the function signature like 
```cuda
// function 
void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *userData);  
```

## Understand CUDA's Dynamic Parallelism 
### What's CUDA dynamic parallelism(CDP)
```
CDP is a device runtime feature that enables nested calls from device functions. 
These nested calls allow different parallelism for the child grid. (Only Tesla and Volta support this).
CDP is useful when you need a different block size depending on the problem.s
```

## Understand OpenMP 
### What's OpenMP 
```
OpenMP(Open Multi-Processing) is a share-memory parallel programming API that can be adopted in multi-platforms(C, C++ and Fortran).
It provides a simple and flexible coding pattern which allows developers involved parallel logic codes into their own projects to help
them enhance the compute efficiency.
```

### What are the OpenMP's Features?
```
1. Simple coding pattern
2. Multi-language supports
3. Shared-Memory model: OpenMP adopts shared memory model that all threads share a same address space that allows 
threads data sharing and communciation more simple. 
4. Provide both Parallel Regions and Work-Sharing to allocate and schedule different task. And allows mult-threads can be execute the code blocks in parallel. 
5. Dynamic Thread Management: OpenMP allows dynamic thread number manage. Threads allocated to tasks can be adjusted dynamically according to system resource. 
```

### OpenMP Basic Usage
1. Parallel Region Declaration 
```cpp
#include <omp.h>
#include <stdio.h>

int main() {
    // declare parallel region by follwing macro instruction
    #pragma omp parallel
    {
        printf("Hello world "); 
    }

    return 0; 
}
```

2. Working Sharing
WOrking sharing struct enables working loading dispatches to multi-threads.
Common instructions like `for`, `sections` and `single`

```c
#include <omp.h>
#include <stdio.h>

int main() {
    int i, n = 10; 
    int a[10], b[10] , sum[10]; 

    // init arr
    init_arr(a,b,c); 

    #pragma omp parallel for
    for (inti = 0; i < n; i++) {
        sum[i] = a[i] + b[i]; 
    }

    print_resultsum(); 
}
```

3. sync strategy 
OpenMP provides multiple sync strategies to avoid data race and maintain thread safety. 
Common instructions like `critical`, `atomic` and `barrier`.

```c
#include <omp.h>
#include <stdio.h>

int main() 
{
    int sum = 0; 
    #pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        #pragma omp atomic
        sum += i; 
    }

    show result
    return 0; 
}
```

### What's the relationship between OpenMP and CUDA
OpenMP and CUDA are both used to accelerate computational tasks through parallelism, but they 
are designed for different types of parallel computing env.

OpenMP is often used for parallel programming on shared-memory architectures(like multi-core CPUs),
while CUDA is used for parallel programming on NVIDIA GPUs, which are specialized for massive parallelism with 
throusands of cores.  

1. (OpenMP and CUDA) Complementary Use: 
OpenMP and CUDA can be used together in a complementary manner. 
While CUDA handles the parallelism on the GPU, OpenMP can handle parallelism on the CPU.
This combination can be particularly powerful for hybrid systems where both the CPU and GPU 
are used to perform computations. 

2. Different Levels of Parallelism:
OpenMP is suitable for coarse parallelism typically found in multi-threaded CPU applications,
whereas CUDA is suitable for fine-grained parallelism on GPUs. 

3. Data Transfer Management: 
When using both OpenMP and CUDA, managing data transfer between CPU and GPU becomes crucial.
Data needs to be efficiently moved between the host(CPU) memory and the device (GPU) memory.

### What kind of scenatrios will OpenMP and CUDA be used together? They often adopted to solve what kind of problems?
Combining OpenMP and CUDA is useful for solving problems that require the strengths of 
both CPU and GPU parallelism. Here are some common scenarios:

1. Heterogeneous Computing.
2. Load Balancing:
For workloads that can be decomposed into tasks with varying computational requirements, OpenMP can be used to balance 
the load between the CPU and GPU. 
3. Data-Parallel and Task-Parallel: 
Combining OpenMP and CUDA allows developers to exploit both data-parallelism on GPUs and task-parallelism on CPUs. 




## Understand MPI
### Install MPI Library on Linux 
```
#!/bin/bash
MPI_VERSION="3.0.4"

wget -O /tmp/openmpi-${MPI_VERSION}.tar.gz https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-${MPI_VERSION}.tar.gz
tar xzf /tmp/openmpi-${MPI_VERSION}.tar.gz -C /tmp
cd /tmp/openmpi-${MPI_VERSION}
./configure --enable-orterun-prefix-by-default
make -j $(nproc) all && sudo make install
sudo ldconfig
mpirun --version
```
### MPI, OpenMP and CUDA 
MPI(Message Passing Interface), OpenMP(Open Multi-Processing) and CUDA(Compute Unified Device Architecture) are three common parallel programming technologies. 
#### MPI
```
* usage: 
multiple && independent compute notes's communication.

* how to work: 
message oriented, each process has its own isolated space, exchange data middle results via messaeg. 

* classical scenrios: 
large scale distributed computing env
```

### OpenMP (Open Multi-Processing)
```
* usage:
parallel computing via shared-memory

* how to work:
multi-thread based. developer defines the range of parallel region via pragma 
compiler recognizes pragma then generate multi-threads' based
```

### CUDA(Compute Unified Device Architecture)
```
* usage:
executes upon nvidia gpu

* how to work:
single instruction multiple threads. developer defines kernel function will be executed by multiple gpu-based threads. 
```
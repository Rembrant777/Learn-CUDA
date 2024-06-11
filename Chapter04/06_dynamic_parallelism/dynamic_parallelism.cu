#include <cstdio>
#include <cstdlib>
#include <iostream>

/**
 dynamic_parallelism.cu cannot be compiled as expected  TAT/~ 

 /usr/local/cuda-9.2/bin/nvcc -ccbin g++ -I/usr/local/cuda-9.2/samples/common/inc   -o dynamic_parallelism.out dynamic_parallelism.cu
 dynamic_parallelism.cu(44): error: calling a __global__ function("child_kernel") from a __global__ function("parent_kernel") is only allowed on the compute_35 architecture or above

 The above error message tells us that our GPU's architecture does not support 

 dynamic parallelism
*/

using namespace std; 

// BUF_SIZE = 2^10 = 1024 
#define BUF_SIZE (1 << 10)
#define BLOCKDIM 256

__global__ void child_kernel(int *data, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&data[idx], seed);
}

__global__ void parent_kernel(int *data)
{
    if (threadIdx.x == 0)
    {
        int child_size = BUF_SIZE/gridDim.x;
        child_kernel<<< child_size/BLOCKDIM, BLOCKDIM >>>(&data[child_size*blockIdx.x], blockIdx.x+1);
    }
    // synchronization for other parent's kernel output
    cudaDeviceSynchronize();
}

// parent_kernel<<block = 2, thread_per_block = 1>>> =>
// parent-block-1: => block-1#threadIdx = 0 invoke child threads via child_kernel<<<a,b>>> 
// parent-block-2: => block-2#threadIdx = 0  invokes child threads via child_kernel<<<a,b>>>
int main()
{
    int *data; 
    int num_child = 2; 

    cudaMallocManaged((void**)&data, BUF_SIZE * sizeof(int)); 
    cudaMemset(data, 0, BUF_SIZE * sizeof(int)); 

    // here create two blocks each block allocate 1 thread to execute 
    // parent_kernel
    // and in parent_kernel only block(i)#threadIdx = 0(the top rank thread) 
    // is allowed to create blocks to execute the inner child_kernel function via gpu resources.
    parent_kernel<<<num_child, 1>>>(data); 

    cudaDeviceSynchronize(); 

    // count elements value 
    int counter = 0; 
    for (int i = 0; i < BUF_SIZE; i++) {
        counter += data[i]; 
    }

    // getting answer 
    int counter_h = 0; 
    for (int i = 0; i < BUF_SIZE; i++) {
        counter_h += (i + 1); 
    }
    // counter_h *= 512 / 2 
    counter_h *= BUF_SIZE / num_child; 

    if (counter_h == counter) {
        printf("Correct !!\n"); 
    } else {
        printf("Error !! Obtained %d. It should be %d\n", counter, counter_h); 
    }

    cudaFree(data); 

    return 0; 
}
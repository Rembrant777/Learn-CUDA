#include <stdio.h>
#include "reduction.h"

/*
  Parallel sum reduction using shared memory
  - takes log(n) steps for n input elements
  - uses n threads 
  - only works for power-of-2 arrays 
*/

__global__ 
void reduction_kernel_2(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; 

    // s_data array size already applied via fuction's 3rd parameter value 
    //   n_threads * sizeof(float) each thread will be allocated by a s_data with length = 1 * sizeof(float)
    extern __shared__ float s_data[]; 

    s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f; 

    __syncthreads(); 

    // do reduction 
    // sequential addressing
    // threadIdx range [0, blockDim.x -1]
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride]; 
        }
        __syncthreads(); 
    }

    // even though the upper operaiton is executed by multiple 
    // threads from block grained scope upon the block grained shared memory
    // but the below code logic only allowed to be execute by one thread that is with the thread
    // id = 0 
    // via this if condition, write conflict between block grained shared memory to global memory can be avoid
    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0]; 
    }
}

int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int n_blocks = (size + n_threads - 1) / n_threads; 
    reduction_kernel_2<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size); 
    return n_blocks; 
}
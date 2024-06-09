#include <stdio.h>
#include "reduction.h"

/**
   Parallel sum reduction using shared memory.
   - takes log(n) steps for n input elements. 
   - uses n threads.
   - only works for power-of-2 arrays
*/

// cuda thread synchronization
__global__ 
void reduction_kernel_1(float *g_in, float *g_out, unsigned int size)
{
    // idx_x is current thread's global(block) scope init offset index value
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; 

    // __shared__ decorated float array means 
    // apply shared memory from block scope to store middle result
    extern __shared__ float s_data[]; 

    // copy thread idx accessible ranged data from global input array 
    // to local shared memory 
    s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f; 

    // waiting all threads copy ready
    __syncthreads(); 

    // do reduction 
    // interleaved addressing

    // this is called the tree-based reduction 
    // each thread get access to different partition of the global memory data
    // and execute the comput logic
    // set n = total threads per block 
    // this algorithm's time complexity is O(log n), current_step_length = 2* previous_step_length
    // and total iteration cnt = log2(n)


    /**
     Step-1 (stride = 1)：
        threadIdx.x = 0 的线程：s_data[0] += s_data[1]
        threadIdx.x = 1 的线程：s_data[2] += s_data[3]
        threadIdx.x = 2 的线程：s_data[4] += s_data[5]
        threadIdx.x = 3 的线程：s_data[6] += s_data[7]

    Step-2 (stride = 2 = stride' *= 2)：
        threadIdx.x = 0 的线程：s_data[0] += s_data[2]
        threadIdx.x = 1 的线程：s_data[4] += s_data[6]

    Step-3 (stride = 4 = stride' *= 2)：
        threadIdx.x = 0 的线程：s_data[0] += s_data[4]
    */
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * threadIdx.x; 
        
        if (index < blockDim.x) {
            s_data[index] += s_data[index + stride]; 
        }

        __syncthreads(); 
    }


    // only one thread that with the id = 0
    // is permitted to write data from local shared memory to global memory 
    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0]; 
    }
}

int reduction(float *g_outputPtr, float *g_inputPtr, int size, int n_threads) 
{
    int n_blocks = (size + n_threads - 1) / n_threads; 
    reduction_kernel_1<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outputPtr, g_inputPtr, size); 
    return n_blocks; 
}



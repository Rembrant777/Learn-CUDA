#include <stdio.h>
#include "reduction.h"

/**
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - use n threads 
    - only works for power-of-2 arrays
*/
__global__
void reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; 
    extern __shared__ float s_data[]; 

    // accumulates input with grid-stride loop and save to shared memory 
    // blockDim.x = how many threads applied each block to execute reduction_kernel function 
    // gridDim.x = how many blocks applied to execute current reduction_kernel function 
    float input = 0.f; 
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x) {
        input += g_in[i]; 
    }
    s_data[threadIdx.x] = input; 

    __syncthreads(); 

    // do reduction 
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride]; 
        }
        __syncthreads(); 
    }

    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0]; 
    }
}

int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int num_sms; 
    int num_blocks_per_sm; 
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0); 
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, 
                    n_threads, n_threads * sizeof(float)); 
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);                 
    
    reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size); 
    reduction_kernel<<<1, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, n_blocks);  

    return 1; 
}


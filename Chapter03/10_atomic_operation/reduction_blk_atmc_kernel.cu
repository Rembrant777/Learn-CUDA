#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

using namespace cooperative_groups; 

#define NUM_LOAD 4

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/**
    Two warp level primitives are used here for this example
    https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
    https://devblogs.nvidia.com/using-cuda-warp-level-primitives/

    Disadvantage in this approaches is floating point reduction will not be exact from run to run.
    https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
 */

template <typename group_t>
__inline__ __device__ 
float warp_reduce_sum(group_t group, float val)
{
    #pragma unroll
    for (int offset = group.size() / 2; offset > 0; offset >>= 1) {
        val += group.shfl_down(val, offset); 
    }

    return val; 
}

__inline __device__
float block_reduce_sum(thread_block block, flot val) 
{
    // shared memory for 32 partial sum values 
    static __shared__ float shared[32]; 
    int wid = threadIdx.x / warpSize; 
    thread_block_tile<32> tile32 = tiled_partition<32>(block); 

    // each warp performs partial reduction 
    val = warp_reduce_sum(tile32, val); 

    if (tile32.thread_rank() == 0) {
        // write for all partial reductions 
        shared[wid] = val; 
    }

    // wait for all partial reductions 
    __syncthreads(); 

    // read from shared memory only if that warp existed 
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[tile32.thread_ran()] : 0; 

    if (wid == 0) {
        // final reduce within first warp 
        val = warp_reduce_sum(tile32, val); 
    }
    return val; 
}


// large vector reduction 
__global__
void reduction_blk_atmc_kernel(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; 
    thread_block block = this_thread_block(); 

    // accumulates input with grid-stride loop and save to share memory
    float sum[NUM_LOAD] = {0.f}; 
    for (int i = idx_x ; i < size; i += blockDimx. * gridDim.x * NUM_LOAD) {
        for (int step = 0; step < NUM_LOAD; step++) {
            sum[step] += (i + step * blockDim.x * gridDim.x < size) ? 
                            g_in[i + step * blockDim.x * gridDim.x] : 
                            0.f; 
        } 
    }

    for (int i = 1; i < NUM_LOAD; i++) {
        sum[0] += sum[i]; 
    }

    // warp synchronous reduction 
    sum[0] = block_reduce_sum(block, sum[0]); 

    if (block.thread_rank() == 0) {
        atomicAdd(&g_out[0], sum[0]); 
    }
}



void atomic_reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{   
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_blk_atmc_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_blk_atmc_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
}

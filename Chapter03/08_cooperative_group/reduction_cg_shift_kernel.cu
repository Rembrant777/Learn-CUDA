#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

using namespace cooperative_groups; 

#define NUM_LOAD 4

/**
    Parallel sum reduction using shared memory 
    - takes log(n) steps for n inupt elements
    - uses n threads 
    - only works for power-of-2 arrays
*/

/**
    Two warp level primitives are used here for this example
    https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
    https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
 */

// __device__ decorated function only invoked by GPU side 
// which means only function decorated by __global__ or __device__ can invoke this function 
 template <typename group_t>
 __inline__ __device__ float warp_reduce_sum(group_t group, float val)
 {
    for (int offset = group.size() / 2; offset > 0; offset >>=1 ) {
        val += group.shfl_down(val, offset); 
    }
    return val; 
 }

 __inline__ __device__ float block_reduce_sum(thread_block block, float val) 
 {
    // apply shared memory to store 32 partial sums 
    __shared__ float shared[32]; 

    // use block scoped thread index / total warpSize = current warp index in the block scope
    int warp_idx = block.thread_index().x / warpSize; 

    // partial reduction at tile<32> side
    // here apply 32 threads from the block to create a tile32
    // tile32 is the code layer's warp, CUDA do not have any struct defined for warp 
    // only tile* which define how many threads in a warp like 
    // thread_block_tile<64> has 64 threads from the block
    // actually thread_block_tile<NUM> tile-NUM = tiled_partition<max(block.max.thread, NUM)>(block); 
    thread_block_tile<32> tile32 = tiled_partition<32>(block); 
    val = warp_reduce_sum(tile32, val); 

    // write reduced value to shared memory 
    // same story only index of thread 1 in the current scop (warp represents in tile32 in CUDA)
    // is allowed to do the write operation 
    if (tile32.thread_rank() == 0) {
        shared[warp_idx] = val;  
    }

    // Wait for all partial reductions 
    block.sync(); 

    // read from shared memory only if it is the leading warp 
    // block.group_dim().x means how many threads available in current block 
    if (warp_idx == 0) {
        val = (threadIdx.x < block.group_dim().x / warpSize) ? shared[tile32.thread_rank()] : 0; 
        // Finally, reduce within first warp 
        val = warp_reduce_sum(tile32, val); 
    }
    return val; 
 }

 // cuda thread synchronziation 
 __global__
 void reduction_kernel(float *g_out, float *g_in, unsigned int size)
 {
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; 
    thread_block block = this_thread_block(); 

    // accumulate input with grid-stride loop and save to the shared memory 
    float sum[NUM_LOAD] = {0.f}; 

    for (int i = idx_x; i < size; i += block.group_dim().x * gridDim.x * NUM_LOAD) {
        for (int step = 0; step < NUM_LOAD; step++) {
            sum[step] += 
                (i + step * block.group_dim().x * gridDim.x < size) ?
                g_in[ i + step * block.group_dim().x * gridDim.x] :
                0.f; 
        }

        for (int i = 1; i < NUM_LOAD; i++) {
            sum[0] += sum[i]; 
        }

        // warp synchronous reduction 
        sum[0] = block_reduce_sum(block, sum[0]); 

        if (block.thread_index().x == 0) {
            g_out[block.group_index().x] = sum[0]; 
        }
    }

 }


 void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
    reduction_kernel<<< 1, n_threads >>>(g_outPtr, g_outPtr, n_blocks);
}
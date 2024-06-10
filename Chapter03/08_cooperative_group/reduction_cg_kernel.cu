#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

using namespace cooperative_groups; 

#define NUM_LOAD 4

/**
 Parallel sum reduction using shared memory 
 - takes log(n) steps for n input elements
 - uses n threads 
 - only works for power-of-2 arrays
*/

// cuda thread sync
__global__
void reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; 
    thread_block block = this_thread_block(); 
    extern __shared__ float s_data[]; 

    // accumulate input with grid-stride loop and save to share memory
    // block.group_dim().x means the number of threads in the current block
    // gridDim.x means the total number of blocks in the grid
    // NUM_LOAD is a constant that indicate how many elements each thread should handle
    // 
    // then, idx_x means the init offset of the input linear array
    // because this loop is grid-grained, so we need to guarantee that the thread handle data cannot overlap 
    // that is why after current thread handle current data, it will jump {total block num} * {each block total thread num}
    // because {total block num} (block.group_dim().x) * {each block total num} (gridDim.x) = Grid total threads
    float input[NUM_LOAD] = {0.f}; 
    for (int i = idx_x; i < size; i += block.group_dim().x * gridDim.x * NUM_LOAD) {
        // NUM_LOAD controls how many elements to handle in each iterate 
        // NUM_LOAD controls inner loop 
        for (int step = 0; step < NUM_LOAD; step++) {
                           // make sure index in range of the [0, size-1]
            input[step] += (i + step * block.group_dim().x * gridDim.x < size) ? 
                            // if index is valid that is in the range of [0, size-1]
                            // retrieve data from global memory to local shared memory to compute(accumulate)
                        g_in[i + step * block.group_dim().x * gridDim.x] :
                            // otherwise, set 0 to it 
                        0.f; 
        }
        // accumulate middle accumulate results to input[0] = input[1] + ... + input[NUM_LOAD_1]
        for (int i = 1; i < NUM_LOAD; i++) {
            input[0] += input[i]; 
        }

        // set current iteration's middle result to global result 
        s_data[threadIdx.x] = input[0]; 
        
        // block grained threads sync  
        block.sync(); 

        // do reduction 

        // stride init value  = block total threads / 2
        // stride every time divide by 2 
        // index = current thread index + s_data[current_thread_index + stride]
        for (unsigned int stride = block.group_dim().x / 2; stride > 0; stride >>= 1) {
            // scheduled threads reduce for every iteration 
            // and will be smaller than a warp size (32) eventually
            if (block.thread_index().x < stride) {
                s_data[block.thread_index().x] += s_data[block.thread_index().x + stride]; 
            }
            block.sync(); 
        }

        if (block.thread_index().x == 0) {
            g_out[block.group_index().x] = s_data[0]; 
        }
    }
}

void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    // num_sms: how many stream multi-processor GPU has in total 
    int num_sms;

    // num_blocks_per_sm: how many blocks each sm has 
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, 
                reduction_kernel, n_threads, n_threads*sizeof(float));

    // total block apply to execute the compute = 
    // min(max num of blocks that gpu can provide, total compute size / number of threads)
    // depends on the data scale 
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_kernel<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_inPtr, size);
    reduction_kernel<<< 1, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_outPtr, n_blocks);
}
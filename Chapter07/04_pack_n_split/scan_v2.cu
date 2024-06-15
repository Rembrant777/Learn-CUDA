#include "scan.h"

__global__
void scan_v2_kernel(float *d_output, float *d_input, int length)
{
    // each threads from same block will 
    // first, retrieve current threadIdx.x and calculate the global index idx in the global input array 
    // second, retrieve corresponding global value via its global index from the array 
    // third, cache the global value to local shared_memory(shared range: all threads from the same block)
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    int tid = threadIdx.x; 

    extern __shared__ float s_buffer[]; 
    s_buffer[threadIdx.x] = d_input[idx]; 
    s_buffer[threadIdx.x + BLOCK_DIM] = d_input[idx + BLOCK_DIM]; 

    int offset = 1; 

    // interesting here
    // first create two index idx_a, idx_b 
    // while loop controls 
    // 1. iterate times = length / 2^stride, 
    // 2. stride = offset * 2 (offset <<= 1)
    
    // why there is (2 * tid + 1) this operation? 
    // the aim to do this is to avoid thread's access index conflict in shared memory range 

    // for example a[0... 8], and current threadIdx.x = 2 
    // loop-1:
            // length = 8, offset = 1
            // idx_a = 1 + (2 * tid + 1) - 1 = 1 + (2 * 2 + 1) -1 = 5
            // idx_b = 1 + (2 * tid + 2) - 1 = 1 + (2 * 2 + 2) -1 = 6
            // then exeucte range validate operation, then accumulate value to {s_b[idx_a] + s_b[idx_b] -> s_b[idx_b]}
            // s_buffer[6] = s_buffer[6] + s_buffer[5] 
    while (offset < length) {
        __syncthreads(); 

        int idx_a = offset * (2 * tid + 1) - 1; 
        int idx_b = offset * (2 * tid + 2) - 1; 

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM) {
#if (DEBUG_INDEX > 0)
            printf("[%d %d]\t", idx_a, idx_b); 
#endif             
            s_buffer[idx_b] += s_buffer[idx_a]; 
        }

        // offset move forward offset * 2 steps 
        offset <<= 1; 
#if (DEBUG_INDEX > 0)
        if (tid == 0) printf("\n-----------------------------------------\n"); 
#endif      
    }

    offset >>= 1; 
    while (offset > 0) {

        __syncthreads(); 

        int idx_a = offset * (2 * tid + 2) - 1; 
        int idx_b = offset * (2 * tid + 3) - 1; 

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM) {
#if (DEBUG_INDEX > 0)
            printf("[%d %d]\t", idx_a, idx_b); 
#endif             
            s_buffer[idx_b] += s_buffer[idx_a]; 
        }
    offset >>= 1; 
#if (DEBUG_INDEX > 0)
        if (tid == 0) printf("\n-----------------------------------------\n");
#endif 
    }
    __syncthreads(); 

    d_output[idx] = s_buffer[tid]; 
    d_output[idx + BLOCK_DIM] = s_buffer[tid + BLOCK_DIM]; 
}

void scan_v2(float *d_output, float *d_input, int length) {
    dim3 dimBlock(BLOCK_DIM); 
    dim3 dimGrid((length + (2 * BLOCK_DIM) -1) / (2 * BLOCK_DIM)); 
    scan_v2_kernel<<<dimGrid, dimBlock, sizeof(float) * BLOCK_DIM * 2 >>>(d_output, d_input, length); 
    cudaDeviceSynchronize(); 
}

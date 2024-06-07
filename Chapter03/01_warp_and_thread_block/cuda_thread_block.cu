#include <stdio.h>
#include <stdlib.h>

/**
In this section, we will discover concurrent operation in CUDA
1) blocks in grid: concurrent tasks, no guarantee their order of execution(no synchronization)
2) warp in blocks: concurrent threads, explicitly synchronizable (this will be discussed in next section)
3) thread in warp: implicitly synchronized
*/

__global__ 
void idx_print() 
{
    // linearized index calculation  
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    // warp index current thread belong
    // warp index = current thread id / total warp count 
    int warp_idx = threadIdx.x / warpSize; 

    // lane index is index of thread defined in the scope of warp 
    int lane_idx = threadIdx.x & (warpSize - 1); 

    if ((lane_idx & (warpSize / 2 - 1)) == 0) {
        // thread, block, warp, lane
        printf(" %5d\t%5d\t %2d\t%2d\n", idx, blockIdx.x, warp_idx, lane_idx);
    }
}

int main(int argc, char* argv[]) 
{
    if (argc == 1) {
        puts("Please put Block Size and Thread Block Size..");
        puts("./cuda_thread_block [grid size] [block size]");
        puts("e.g.) ./cuda_thread_block 4 128");

        exit(1);
    }

    int gridSize = atoi(argv[1]); 
    int blockSize = atoi(argv[2]);

    // here we try the test sets

    // test case-1:
    // command ./cuda_thread_block 4 128
    // input: grid = 4(this means how many blocks we apply), block = 128(this means how many threads per block)
    // deduce: warp unit = 32(how many thread per warp), warp count = 128 / 32 = 4
    // warp id range [0 ... 3] 
    // block id range [0 ... 3]
    // thread id is generated dynamically via gpu which cannot be deduced 
    // lane id is calculated from thread id also cannot be deduced

    // test case-2:
    // command ./cuda_thread_block 12 256 
    // input: grid = 12 (how many block we apply), block = 256 (how many threads per block)
    // deduce: warp unit = 32 (how many thread per warp), warp count = 256 / 32 = 8
    // warp id range [0 ... 7]
    // block id range [0 ... 11]
    puts("thread, block, warp, lane");
    idx_print<<<gridSize, blockSize>>>(); 
    cudaDeviceSynchronize();   
}
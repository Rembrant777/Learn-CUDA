#include <iostream>
#include <math.h>

// CUDA kernel to add elements of two arrays
////////////////////////////////////////////////////////////////////////////////////////////////
// we all know that function below is the kernel function 
// kernel function will be executed by each (cuda)thread from different (cuda) blocks in parallel
// 
// blockId.x: current thread locates in which block id 
// blockDim.x: how many threads in total allocated from each block 
// threadId.x: thread's index in each block 
// index: current thread handle linear array initialized offset
// stride: how many threads in total allocated from the grid which can be calcuated by 
// = total block num * threads per block 
////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 

    // so here we start index is current thread's init offset of the linear array
    // and each step jump total threads counts 
    // which = total blocks on x dimension (blockDim.x) each block * total blocks applied on x dimesnion(gridDim.x) 

    // let's review the mapping relationship between the thread and the block and the grid 
    // thread is all the basic of gpu's parallel execution 
    // thread:block = 1:N (N >= 1)
    // block:grid = 1:N (N >= 1)
    for (int i = index; i < n; i += stride) {
        // stride: 步长, 这就说得通了
        // 每次线程跳跃处理线性数组中的数据, 不同 block 中的不同线程从目标线性数组中不同位置获取  input 数据加以处理, 然后再写回到缓存中, 
        // block 间的线程内存是彼此隔离的, block 中的线程们的内存是可以共享的存在一定概率写覆盖, 
        // 所以通常在核函数中处理数据后通常会调用 kernel 层级的 __syncthreads 函数来规避 conflict
        // 同步好的数据存放在  block 层级的缓存中, 所有线程执行完成后, 统一往最终汇聚结果的缓存中写入即可
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    // N = 1048576
    int N = 1<<20; 
    float *x, *y; 

    // Allocate Unified Memory -- accessible from CPU or GPU 
    cudaMallocManaged(&x, N * sizeof(float)); 
    cudaMallocManaged(&y, N * sizeof(float)); 

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f; 
        y[i] = 2.0f; 
    }

    // Launch kernel on 1M elements on the GPU 
    int blockSize = 256; 
    int numBlocks = (N + blockSize -1) / blockSize; 
    add<<<numBlocks, blockSize>>>(N, x, y); 

    // wait for GPU to finish before accessing on host 
    cudaDeviceSynchronize(); 

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f; 
    for (int i = 0; i < N; i++) {
        // std::cout << "y[i] = " << y[i] << std::endl; 
        maxError = fmax(maxError, fabs(y[i]-3.0f)); 
    }
    std::cout << "Max error: " << maxError << std::endl; 

    // free cuda buffer
    cudaFree(x); 
    cudaFree(y); 

    return 0; 
}
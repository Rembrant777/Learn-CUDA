#include <iostream>
#include <math.h>

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    // each thread's init offset of to be computed linear array
    int index = blockIdx.x * blockDim.x + threadIdx.x; 

    // total threads count set as each step length
    // total threads count = total blocks * each block's total thread count 
    int stride = blockDim.x * gridDim.x; 
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i]; 
    }
}

int main(void) 
{
    int N = 1 << 20; 
    float *x, *y; 
    int device = -1; 

    // apply unified memory -- which can be accessed on both GPU and CPU side
    cudaMallocManaged(&x, N * sizeof(float)); 
    cudaMallocManaged(&y, N * sizeof(float)); 

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f; 
        y[i] = 2.0f; 
    }

    cudaGetDevice(&device); 
    // GPU prefetches unified memory 
    cudaMemPrefetchAsync(x, N * sizeof(float), device, NULL); 
    cudaMemPrefetchAsync(y, N * sizeof(float), device, NULL); 

    // Launch kernel on 1M elements on the GPU 
    int blockSize = 256; 
    int numBlocks = (N + blockSize - 1) / blockSize; 
    add<<<numBlocks, blockSize>>>(N, x, y); 

    // Host prefetches Memory
    cudaMemoryPrefetchAsync(y, N * sizeof(float), cudaCpuDeviceId, NULL); 

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize(); 

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f; 
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f)); 
    }
    std::cout << "Max error: " << maxError << std::endl; 

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;   
}
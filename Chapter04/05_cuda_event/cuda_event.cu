#include <cstdio>
#include <helper_timer.h>

using namespace std; 

__global__ void vecAdd_kernel(float *c, const float *a, const float *b); 
void init_buffer(float *data, const int size); 

int main(int argc, char *argv[]) 
{
    float *h_a, *h_b, *h_c; 
    float *d_a, *d_b, *d_c; 
    int size = 1 << 24; 
    int bufsize = size * sizeof(float); 

    // allocate host memories 
    cudaMallocHost((void **)&h_a, bufsize); 
    cudaMallocHost((void **)&h_b, bufsize); 
    cudaMallocHost((void **)&h_c, bufsize); 

    // initialize host value 
    srand(2024); 
    init_buffer(h_a, size); 
    init_buffer(h_b, size); 
    init_buffer(h_c, size); 

    // allocate device memories
    cudaMalloc((void **) &d_a, bufsize); 
    cudaMalloc((void **) &d_b, bufsize);
    cudaMalloc((void **) &d_c, bufsize); 

    // copy host -> device 
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice); 

    // initialize the host timer 
    StopWatchInterface *timer; 
    sdkCreateTimer(&timer); 

    // --- here show how to use CUDA Event --- 
    cudaEvent_t start, stop; 
    // create CUDA events 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 

    // start to measure the execution time 
    sdkStartTimer(&timer); 
    cudaEventRecord(start); 

    // here we launch cuda kernel 
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x); 
    vecAdd_kernel<<<dimGrid, dimBlock>>>(d_c, d_a, d_b); 

    // record the event right after the kernel execution finished 
    cudaEventRecord(stop); 

    // synchornize the device to measure the execution time from the host side
    // we also can make synchronization based on CUDA event 
    cudaEventSynchronize(stop); 
    sdkStopTimer(&timer); 

    // copy data from device -> host 
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost); 

    // print out the result 
    int print_idx = 256; 
    printf("compared a sample result ...\n"); 
    printf("host %.6f, device: %.6f\n", h_a[print_idx] + h_b[print_idx], h_c[print_idx]); 

    // print estimated kernel execution time
    // different from previous StopWatchInterface time record
    // this time we use cuda event start, stop to record the cuda function execute time 
    float elapsed_time_msed = 0.f; 
    cudaEventElapsedTime(&elapsed_time_msed, start, stop); 
    printf("CUDA event estimated(CUDA Event Recorded) - elapsed %.3f ms\n", elapsed_time_msed); 

    // compute and print the perforamce 
    elapsed_time_msed = sdkGetTimerValue(&timer);
    printf("Host measured time = %.3f msed/s \n", elapsed_time_msed); 

    // terminate device memories 
    cudaFree(d_a); 
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // delete timer
    sdkDeleteTimer(&timer);

    // terminate CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// same the old story
// this is the minmum exeuctable code fraciton of CUDA's thread 
__global__ 
void vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

// gpu buffer init function 
void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}
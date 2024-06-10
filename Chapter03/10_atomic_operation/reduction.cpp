#include <stdio.h>
#include <stdlib.h>

// cuda runtime 
#include <cuda_runtime.h>
#include <helper_timer.h>

#include "reduction.h"

void run_benchmark(void (*reduce)(float *, float *, int, int),
                    float *d_outPtr, float *d_inPtr, int size); 

void init_input(float *data, int size);                     

float get_cpu_result(float *data, int size); 

//// program main body 
int main(int argc, char *argv []) 
{
    float *h_inPtr; 
    float *d_inPtr, *d_outPtr; 
    unsigned int size = 1 << 24; 
    float result_host, result_gpu; 
    int mode = 0; 

    if (argc > 1 && atoi(argv[1]) == 1) {
        mode = 1; 
    }

    srand(2024); 

    // allocate memory 
    h_inPtr = (float *) malloc(size * sizeof(int)); 

    // data init with random values
    init_input(h_inPtr, size); 

    // prepare GPU resource 
    cudaMalloc((void **)& d_inPtr, size * sizeof(float)); 
    cudaMalloc((void **)&d_outPtr, size * sizeof(float)); 

    // copy random init data from h_inPtr to d_inPtr 
    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice); 

    // get reduction result from GPU 
    run_benchmark(atomic_reduction, d_outPtr, d_inPtr, size); 
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost); 

    // get all sum from CPU 
    result_host = get_cpu_result(h_inPtr, size); 
    printf("host: %f, device: %f\n", result_host, result_gpu); 

    // terminates memory 
    cudaFree(d_outPtr); 
    cudaFree(d_inPtr); 
    free(h_inPtr); 

    return 0; 
}

void run_reduction(void (*reduce)(float *, float *, int , int),
                    float *d_outPtr, float *d_inPtr, int size, int n_threads) 
{
    reduce(d_outPtr, d_inPtr, size, n_threads); 
}

void run_benchmark(void (*reduce)(float *, float *, int, int),
            float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256; 
    int test_iter = 100; 

    // warm-up
    reduce(d_outPtr, d_inPtr, size, num_threads); 

    // initialize timer 
    StopWatchInterface *timer; 
    sdkCreateTimer(&timer); 
    sdkStartTimer(&timer);

    // operation body 
    for (int i = 0; i < test_iter; i++)  {
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice); 
        run_reduction(reduce, d_outPtr, d_outPtr, size, num_threads); 
    }

    // getting elasped time
    cudaDeviceSynchronize(); 
    sdkStopTimer(&timer); 

    // compute and print the performance 
    float elapsed_time_msed = sdkGetTimerValue(&timer) / (float) test_iter; 
    float bandwidth = size * sizeof(float) / elapsed_time_msed / 1e6; 
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);
}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}   
            
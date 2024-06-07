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

////////////////////////////////////////////////////////////////////////////////
// Program Main Entry Point
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) 
{
    float *h_inPtr; 
    float *d_inPtr, *d_outPtr; 
    unsigned int size = 1 << 24; 
    float result_host, result_gpu; 
    srand(2019);

    // Allocate memory
    h_inPtr = (float *) malloc(size * sizeof(float));  

    // Data initialization with random values
    init_input(h_inPtr, size); 

    // Prepare GPU resource
    cudaMalloc((void **) &d_inPtr, size * sizeof(float)); 
    cudaMalloc((void **) &d_outPtr, size * sizeof(float)); 

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice); 

    // Get reduction result from GPU 
    run_benchmark(global_reduction, d_outPtr, d_inPtr, size); 
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost); 

    // Get reduction result from GPU
    // Get all sum from CPU
    result_host = get_cpu_result(h_inPtr, size); 
    printf("host: %f, device: %f\n", result_host, result_gpu); 

    // Terminates memory
    cudaFree(d_outPtr); 
    cudaFree(d_inPtr); 
    free(h_inPtr); 

    return 0; 
}
#include <stdio.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>
#include <helper_timer.h>
#include "reeduction.h"

void run_benchmark(void (*reduce)(float *, float *, int, int),
                    float *d_outPtr, float *d_inPtr, int size);
void init_input(float *data, int size);  
float get_cpu_result(float *data, int size); 

int main(int argc, char *argv []) 
{
    float *h_inPtr; 
    float *d_inPtr, *d_outPtr; 
    unsigned int size = 1 << 24; 
    float result_host, result_gpu; 
    srand(2024);

    // allocate memory 
    h_inPtr = (float*) malloc(size * sizeof(float));  

    // data initialization with random values
    init_input(h_inPtr, size); 

    // prepare GPU resource 
    cudaMalloc((void **)&d_inPtr, size * sizeof(float)); 
    cudaMalloc((void **)&d_outPtr, size * sizeof(float)); 

    cudaMemory(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice); 

    // get reduction result from GPU 
    run_benchmark(reduction, d_outPtr, d_inPtr, size); 
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost); 

    // get reduction result from GPU 

    // get all sum from CPU 
    result_host = get_cpu_result(h_inPtr, size);
    printf("host: %f, device: %f\n", result_host, result_gpu); 

    // terminates memory 
    cudaFree(d_outPtr); 
    cudaFree(d_inPtr); 
    free(h_inPtr); 

    return 0; 
}
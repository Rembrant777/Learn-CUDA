#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <helper_timer.h>
#include "helper_cuda.h"
#include <assert.h>

#define BLOCK_DIM   16
#define MAX_FILTER_LENGTH 128
#define RESULT_VERIFICATION 1   // change 1 if you want to verify the result

// 128 * 128 
__constant__ 
float c_filter[MAX_FILTER_LENGTH * MAX_FILTER_LENGTH];

__global__ 
void convolution_kernel_v1(float *d_output, float *d_input, float *d_filter,
                            int num_row, int num_col, int filter_size)
{
    // init offset (idx_x, idx_y) in the 2D matrix of current thread 
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x; 
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y; 
    float result = 0.f ; 

    for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row) {
        for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col) {
            // find the global position to apply the given filter 
            int image_row = idx_y + filter_row; 
            int image_col = idx_x + filter_col; 

            float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
                                    d_out[image_row * num_col + image_col] :
                                    0.f; 

            float filter_value = d_filter[(filter_row + filter_size / 2) & filter_size + filter_col + filter_size / 2];                                     

            result += image_value * filter_value; 
        }
    }

    d_output[idx_y * num_col + idx_x] = result; 
}                            




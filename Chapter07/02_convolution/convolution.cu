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

// 128 * 128 
// this __constant__ decorated variable's space 
// is applied from the constant memory space which means this shared_memory is shared among all blocks that applied to 
// exeucte the convolution 

// and to copy data from host(cpu side) to device(gpu side) we need to use the cudaMemcpyToSymbol
// and there is something more I have to write about the constant space
// constant space/memory applied matrix can be accessed by all threads from all blocks 
// its values cannot be overwrite, because it constant 
// it is constant cache which can improve the access speed in some degree, but not fine-grained 
// there is another fine-grained solution which provide the filter kernel in shared_memories 
// this method is implemented in convolution_kernel_v3 method 

// V2: use Block Grained constant cache to hold kernel filter array to improve the access speed
__global__
void convolution_kernel_v2(float *d_output, float *d_input, 
                            float *d_filter, int num_row, int num_col, int filter_size)
{
    // each thread has its own unique global index, this is the unique index in X axis
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x; 

    // each thread has its own unique global index, this is the unique index in Y axis 
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y; 

    float result = 0.f; 

    // filter_size here is the length of the filter array 
    // and both filter_row and filter_col visit filter array range is [-filter_size / 2, filter_size /2]
    for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row) {
        for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col) {

            // image_row and image_col mean on which row and column to retrieve the input image value which is the d_input
            int image_row = idx_y + filter_row; 
            int image_col = idx_x + filter_col; 

            // A ? B : 0.f 
            // in A validate the visit index of (x, y) is in the range of the input image array 
            // in B retrieve correspoinding data from d_input(image), and d_input is a 1D array, 
            //                  so get the value by index_x * len(column) + index_y 
            // in C, if retrieved index not in the valid range of input image
            //                  then get value as 0.f --> this is also be regareded as the padding
            float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
                                    d_input[image_row * num_col + image_col] : 
                                    0.f; 

            // we already know that c_filter is the so called 'kernel filter' with small scal 
            // each element in the 'kernel filter' is called the filter co-efficient
            // this step is retrieving correspoinding filter co-efficient from kernel filter
            // different thread has different thread index value 
            // so different threads exeucte in parallel get access to different regions from the constant kernel filter
            float filter_value = c_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];                                                            

            // use original image_value * filter_value and results will be accumulated to the local variable result 
            result += image_value * filter_value; 
        }
    }

    // finally the accumulated result will be saved to the output array(output image)
    d_output[idx_y * num_col + idx_x] = result; 
}                            


// V3: use Thread Grained Shared Memroy to cache the kernel filter (filter data array)
// to improve the convolution compute speed 

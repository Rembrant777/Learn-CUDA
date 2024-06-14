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
                                    d_output[image_row * num_col + image_col] :
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
__global__
void convolution_kernel_v3(float *d_output, float *d_input, float *d_filter, 
                    int num_row, int num_col, int filter_size)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x; 
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y; 

    int pad_size = filter_size / 2; 
    int tile_size = BLOCK_DIM + 2 * pad_size; 

    // Block scoped shared memory to cache kernel filter 
    extern __shared__ float s_input[]; 

    for (int row = 0; row <= tile_size / BLOCK_DIM; row++) {
        for (int col = 0; col <= tile_size / BLOCK_DIM; col++) {
            // input data index row  
            int idx_row = idx_y + BLOCK_DIM * row - pad_size; 

            // input data index col 
            int idx_col = idx_x + BLOCK_DIM * col - pad_size; 

            // filter index row 
            int fid_row = threadIdx.y + BLOCK_DIM * row; 

            // filter index col 
            int fid_col = threadIdx.x + BLOCK_DIM * col; 

            if (fid_row >= tile_size || fid_col >= tile_size)
                continue; 
            s_input[tile_size * fid_row + fid_col] = \
                    (idx_row >= 0 && idx_row < num_row && idx_col >= 0 && idx_col < num_col) ?
                    d_input[num_col * idx_row + idx_col] :
                    0.f; 
        }
    }

    __syncthreads(); 
    float result = 0.f;
    for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
    {
        for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
        {
            // Find the global position to apply the given filter            
            int image_row = threadIdx.y + pad_size + filter_row;
            int image_col = threadIdx.x + pad_size + filter_col;

            float image_value  = s_input[tile_size * image_row + image_col];            
            float filter_value = c_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

            result += image_value * filter_value;
        }
    }

    d_output[idx_y * num_col + idx_x] = result;
}                    
void convolution_gpu(int version, float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filter_size)
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((num_col + BLOCK_DIM - 1) / BLOCK_DIM, (num_row + BLOCK_DIM - 1) / BLOCK_DIM);
    if (version == 1)
        convolution_kernel_v1<<<dimGrid, dimBlock>>>(d_output, d_input, d_filter, num_row, num_col, filter_size);
    else if (version == 2) 
        convolution_kernel_v2<<<dimGrid, dimBlock>>>(d_output, d_input, d_filter, num_row, num_col, filter_size);
    else // version == 3
    {
        int shared_mem_size = (2*filter_size+BLOCK_DIM) * (2*filter_size+BLOCK_DIM) * sizeof(float);
        convolution_kernel_v3<<<dimGrid, dimBlock, shared_mem_size, 0 >>>(d_output, d_input, d_filter, num_row, num_col, filter_size);
    }
    
    checkCudaErrors(cudaGetLastError());
}

void convolution_host(float *h_output, float *h_input, float *h_filter, int num_row, int num_col, int filter_size)
{
    //For every pixel in the image
    #pragma omp parallel 
    for (int row = 0; row < (int)num_row; ++row)
    {
        for (int col = 0; col < (int)num_col; ++col)
        {
            float result = 0.f;
            //For every value in the filter around the pixel (c, r)
            for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
            {
                for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
                {
                    // Find the global image position for this filter position
                    int image_row = row + filter_row;
                    int image_col = col + filter_col;

                    float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
                                            h_input[image_row * num_col + image_col] : 0.f;
                    float filter_value = h_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

                    result += image_value * filter_value;
                }
            }

            h_output[row * num_col + col] = result;
        }
    }
}


/* Generates Bi-symetric Gaussian Filter */
void generate_filter(float *h_filter, int filter_size)
{
    float blur_kernel_sigma = 2.;

    float sum_filter = 0.f; //for normalization
    for (int row = -filter_size / 2; row <= filter_size / 2; row++)
    {
        for (int col = -filter_size / 2; col <= filter_size / 2; col++)
        {
            float filterValue = expf(-(float)(col * col + row * row) / (2.f * blur_kernel_sigma * blur_kernel_sigma));
            h_filter[(row + filter_size / 2) * filter_size + col + filter_size / 2] = filterValue;
            sum_filter += filterValue;
        }
    }

    // normalization
    float normalizationFactor = 1.f / sum_filter;
    for (int row = -filter_size / 2; row <= filter_size / 2; row++)
        for (int col = -filter_size / 2; col <= filter_size / 2; col++)
            h_filter[(row + filter_size / 2) * filter_size + col + filter_size / 2] *= normalizationFactor;
}

void generate_data(float *h_buffer, int num_row, int num_col)
{
    for (int row = 0; row < num_row; row++) {
        for (int col = 0; col < num_col; col++) {
            // h_buffer[row * num_col + col] = float(rand() & 0xFFFFFF) / RAND_MAX;
            h_buffer[row * num_col + col] = 1.f;
        }
    }
}

bool value_test(float *a, float *b, int length)
{
    float epsilon = 0.000001;
    bool result = true;
    for (int i = 0; i < length; i++)
        if (abs(a[i] - b[i]) >= epsilon)
            result = false;
    return result;
}

int main()
{
    int num_row = 2048;
    int num_col = 2048;
    int filter_size = 9;
    int buf_size = num_row * num_col * sizeof(float);

    float *h_input, *d_input;
    float *h_output_host, *h_output_gpu, *d_output;
    float *h_filter, *d_filter;

    float elapsed_time_gpu;

    // initialize timer
    StopWatchInterface *timer_host, *timer_gpu;
    sdkCreateTimer(&timer_host);
    sdkCreateTimer(&timer_gpu);

    srand(2019);

    // allocate host memories
    h_input = (float *)malloc(buf_size);
    h_output_host = (float *)malloc(buf_size);
    h_output_gpu = (float *)malloc(buf_size);
    h_filter = (float *)malloc(filter_size * filter_size * sizeof(float));

    // allocate gpu memories
    cudaMalloc((void **)&d_input, buf_size);
    cudaMalloc((void **)&d_output, buf_size);
    cudaMalloc((void **)&d_filter, filter_size * filter_size * sizeof(float));

    // generate data
    generate_data(h_input, num_row, num_col);
    generate_filter(h_filter, filter_size);

    // copy input date to gpu
    cudaMemcpy(d_input, h_input, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_filter, h_filter, filter_size * filter_size * sizeof(float));

    // processing in GPU
    sdkStartTimer(&timer_gpu);
    cudaProfilerStart();
    convolution_gpu(1, d_output, d_input, d_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer_gpu);
    elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
    printf("Processing Time (1) -> GPU: %.2f ms\n", elapsed_time_gpu);

    // processing in GPU
    sdkResetTimer(&timer_gpu);
    sdkStartTimer(&timer_gpu);
    convolution_gpu(2, d_output, d_input, d_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer_gpu);
    elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
    printf("Processing Time (2) -> GPU: %.2f ms\n", elapsed_time_gpu);

    // processing in GPU (3)
    sdkResetTimer(&timer_gpu);
    sdkStartTimer(&timer_gpu);
    convolution_gpu(3, d_output, d_input, d_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer_gpu);
    cudaProfilerStop();
    elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
    printf("Processing Time (3) -> GPU: %.2f ms\n", elapsed_time_gpu);

#if (RESULT_VERIFICATION)
    // processing in CPU
    sdkStartTimer(&timer_host);
    convolution_host(h_output_host, h_input, h_filter, num_row, num_col, filter_size);
    sdkStopTimer(&timer_host);

    float elapsed_time_host = sdkGetTimerValue(&timer_host);
    printf("Processing Time -> Host: %.2f ms\n", elapsed_time_host);

    // compare the result
    cudaMemcpy(h_output_gpu, d_output, buf_size, cudaMemcpyDeviceToHost);
    if (value_test(h_output_host, h_output_gpu, num_row * num_col))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
#endif

    // finalize
    free(h_input);
    free(h_output_host);
    free(h_output_gpu);
    free(h_filter);

    return 0;
}

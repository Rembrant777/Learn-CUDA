#include "loss.h"
#include "helper.h"

#include <cassert>
#include <cuda_runtime.h>

using namespace cudl;

/* 
 * https://deepnotes.io/softmax-crossentropy
 * */

CrossEntropyLoss::CrossEntropyLoss()
{
    cudaMalloc((void**)&d_loss_, sizeof(float));
}

CrossEntropyLoss::~CrossEntropyLoss()
{
    if (d_loss_ != nullptr)
        cudaFree(d_loss_);
        d_loss_ = nullptr;

    if (d_workspace_ != nullptr)
        cudaFree(d_workspace_);
}

/**
 function clip is used to clip given floating value in expected range. 

 Function Logic:
 First, fmax(prediction, epsilon) is used to ensure passing prediction is at least epsilon. 
        If prediction is smaller than epsion it will return epsion.
 
 Then, fmin(result, 1.f - epsion) is used to maintain that the final return result at most 1 - epsilon.

 In this way, given prediciton value will be maintained [epsilon, 1 - epsilon] range.

 This function is designed to avoid input prediction value range out of control cause numerical issueds
 associated with exterme value like A/B and B = 0 or B is overflow as float or int or double. 

 @param prediction the floating-point value that needs to be clipped.
 @param epsilon very small positive number that sets the clipping boundaries, default value is 1e-12
*/
__device__ float clip(float prediction, float epsilon=1e-12)
{
    return fmin(fmax(prediction, epsilon), 1.f - epsilon);
}

__global__ void softmax_loss_kernel(float *reduced_loss, float *predict, float *target, 
                        float *workspace, int batch_size, int num_outputs)
{
    int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float s_data[];
    float loss = 0.f;

    // each thread calculate entropy for each data and accumulate to shared memory
    for (int c = 0; c < num_outputs; c++)
        loss += target[batch_idx * num_outputs + c] * logf(predict[batch_idx * num_outputs + c]);
    workspace[batch_idx] = -loss;

    // then, we do reduction the result to calculate loss using 1 thread block
    if (blockIdx.x > 0) return;

    // cumulate workspace data
    s_data[threadIdx.x] = 0.f;
    for (int i = 0; i < batch_size; i += blockDim.x)
    {
        s_data[threadIdx.x] += workspace[threadIdx.x + i];
    }

    __syncthreads();

    // reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x + stride < batch_size)
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        reduced_loss[blockIdx.x] = s_data[0];
    }
}

void CrossEntropyLoss::init_workspace(int batch_size)
{
    if (d_workspace_ == nullptr)
        cudaMalloc((void**)&d_workspace_, sizeof(float) * batch_size);
}

float CrossEntropyLoss::loss(Blob<float> *predict, Blob<float> *target)
{
    int num_sms = get_cuda_dev_num_sms(); 
    int num_blocks_per_sm = get_num_blocks_per_sm(); 
    
    int batch_size = target->n();
    int num_outputs = target->c();

    init_workspace(batch_size);

    #if (DEBUG_LOSS)
    std::cout << "[[ LOSS ]]" << std::endl;
    predict->print("predict", true);
    target->print("target", true);
    #endif // DEBUG_LOSS

    int num_blocks = min(num_blocks_per_sm * num_sms, \
                         (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);
    softmax_loss_kernel<<< num_blocks, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float), 0 >>>
                (d_loss_, predict->cuda(), target->cuda(), d_workspace_, batch_size, num_outputs);
    cudaMemcpy(&h_loss_, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);
    
    // batch mean loss 
    return h_loss_ / float(batch_size);
}

int CrossEntropyLoss::get_cuda_dev_num_sms()  {
    int ret = 0; 
    cudaDeviceGetAttribute(&ret, cudaDevAttrMultiProcessorCount, 0); 
    return ret; 
}

int CrossEntropyLoss::get_num_blocks_per_sm() {
    int ret = 0; 
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ret, softmax_loss_kernel, 
                BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float));
    return ret; 
}



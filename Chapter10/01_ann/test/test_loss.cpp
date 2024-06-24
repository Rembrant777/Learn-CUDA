#include "../src/loss.h"
#include "../src/helper.h"
#include <gtest/gtest.h>

using namespace cudl; 

TEST(TestLoss, TestLossCreateAndDestroy) {
    CrossEntropyLoss* c_loss = new CrossEntropyLoss(); 
    EXPECT_NE(c_loss, nullptr); 
    EXPECT_NE(c_loss->get_d_loss(), nullptr); 
    EXPECT_NE(c_loss->get_h_loss(), 1.f);
    EXPECT_EQ(c_loss->get_d_workspace(), nullptr);  

    int batch_size = 8; 
    c_loss->test_invoke_init_workspace(batch_size); 
    EXPECT_NE(c_loss->get_d_workspace(), nullptr); 

    delete c_loss; 
}

TEST(TestLoss, clipFuncTest) {
    CrossEntropyLoss *c_loss = new CrossEntropyLoss(); 
    EXPECT_NE(c_loss, nullptr);  

    float epsilon = 1e-12; 

    // prediction is within the range 
    EXPECT_EQ(0.5f, c_loss->test_clip(0.5f, epsilon)); 

    // prediction is exactly the epsilon 
    EXPECT_EQ(epsilon, c_loss->test_clip(epsilon, epsilon)); 

    // prediction is a negative value 
    EXPECT_EQ(epsilon, c_loss->test_clip(-0.5f, epsilon)); 

    // prediction is greater than 1
    EXPECT_EQ(1.f - epsilon, c_loss->test_clip(1.5f, epsilon)); 

    // prediction is 1 
    EXPECT_EQ(1.f - epsilon, c_loss->test_clip(1.0f, epsilon)); 

    // reset epsilon to 1e-6
    float new_epsilon = 1e-6; 
    EXPECT_EQ(0.5f, c_loss->test_clip(0.5f, new_epsilon)); 
    EXPECT_EQ(new_epsilon, c_loss->test_clip(new_epsilon / 2, new_epsilon)); 
    EXPECT_EQ(1.f - new_epsilon, c_loss->test_clip(1.f - new_epsilon / 2, new_epsilon)); 

    delete c_loss; 
}

TEST(TestLoss, testGetCudaSmsNums) {
    CrossEntropyLoss* c_loss = new CrossEntropyLoss(); 
    EXPECT_NE(nullptr, c_loss); 
    int ret = c_loss->get_cuda_dev_num_sms(); 
    EXPECT_NE(ret, 0); 

    ret = c_loss->get_num_blocks_per_sm(); 
    EXPECT_NE(ret, 0); 

    delete c_loss; 
}


__global__ void softmax_loss_kernel_test(float *reduced_loss, float *predict, float *target, 
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
    // here collect each thread's range output value from workspace to shared memory 
    for (int i = 0; i < batch_size; i += blockDim.x)
    {
        s_data[threadIdx.x] += workspace[threadIdx.x + i];
    }

    // wait all threads generates data to shared memory 
    __syncthreads();

    // reduction
    // accumulates via reduction all shared memory to final results 
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

TEST(TestLoss, softmaxLossKernelFuncTest) {
    // create 1 block, each block has two threads.
    // each thread manipulates 3 elements , each thread iterates 3 times to generate 3 output value 
    int batch_size = 2; 
    int num_outputs = 3; 

    // allocate host space to hold predict and target values 
    std::vector<float> h_predict = {0.2f, 0.5f, 0.3f, 0.1f, 0.2f, 0.7f}; 
    std::vector<float> h_target  = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f}; 
    std::vector<float> h_workspace(batch_size); 
    std::vector<float> h_reduced_loss(1); 

    // create pointers on gpu memory spaces 
    float *d_predict, *d_target, *d_workspace, *d_reduced_loss; 

    // allocate space via cuda apis 
    checkCudaErrors(cudaMalloc(&d_predict, h_predict.size() * sizeof(float))); 
    checkCudaErrors(cudaMalloc(&d_target, h_target.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_workspace, h_workspace.size() * sizeof(float))); 
    checkCudaErrors(cudaMalloc(&d_reduced_loss, h_reduced_loss.size() * sizeof(float))); 


    // copy to be calculated data from host to device(gpu)
    checkCudaErrors(cudaMemcpy(d_predict, h_predict.data(), h_predict.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_target, h_target.data(), h_target.size() * sizeof(float), cudaMemcpyHostToDevice));


    // after load datat to gpu space, here we invoke the calculation via gpu kernel api
    int threads_per_block = batch_size; 
    // 1: means we apply 1 block 
    // thread_per_block: each block invoke threads num 
    // thread_per_block * sizeof(float): each block apply shared memory size = thread_cnt * 1 float type bytes
    // which means each block each thread has 1 float byte space
    softmax_loss_kernel_test<<<1, threads_per_block, threads_per_block * sizeof(float)>>>(
            d_reduced_loss, d_predict, d_target, d_workspace, batch_size, num_outputs); 
    checkCudaErrors(cudaMemcpy(h_workspace.data(), d_workspace, h_workspace.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_reduced_loss.data(), d_reduced_loss, h_reduced_loss.size() * sizeof(float), cudaMemcpyDeviceToHost));            

    // here we execute the expected loss value by manual 
    // based on given predict and expected(target) values 
    // since we adopt the cross entropy loss as the loss function 
    // the formular of this function is 
    // https://en.wikipedia.org/wiki/Cross-entropy
    // the first thing is cross-entropy-loss is used upon the probability distribution values.
    // and the second thing is 
    // expected_loss[i] = -sum{target[i] * log(predict[i])}; (i = 1 ... N) 
    // target[batch_idx * num_outputs + c] * logf(predict[batch_idx * num_outputs + c]);
    float expected_loss0 = -(h_target[0] * logf(h_predict[0]) + h_target[1] * logf(h_predict[1]) + h_target[2] * logf(h_predict[2]));
    float expected_loss1 = -(h_target[3] * logf(h_predict[3]) + h_target[4] * logf(h_predict[4]) + h_target[5] * logf(h_predict[5]));
    float expected_reduced_loss = expected_loss0 + expected_loss1; 


    std::cout << "expected_loss0 " << expected_loss0 << ", h_workspace[0] " << h_workspace[0]<< std::endl; 
    std::cout << "expected_loss1 " << expected_loss1 << ", h_workspace[1] " << h_workspace[1]<< std::endl; 

    

    EXPECT_TRUE(fabs(h_workspace[0] - expected_loss0) < 1e-6); 
    EXPECT_TRUE(fabs(h_workspace[1] - expected_loss1) < 1e-6);
    EXPECT_TRUE(fabs(h_reduced_loss[0] - expected_reduced_loss) < 1e-6); 

    // free cuda memory 
    cudaFree(d_predict); 
    cudaFree(d_target); 
    cudaFree(d_workspace); 
    cudaFree(d_reduced_loss);
}

TEST(TestLoss, lossFuncTest) {
    EXPECT_EQ(1, 1); 
}
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cudnn.h>
#include <gtest/gtest.h>

// helper class to wrap CudnnTensor basic operaitons 
class CudnnTensorHelper {
public: 
    CudnnTensorHelper(int n, int c, int h, int w):n_(n), c_(c), h_(h), w_(w) {
        // init cudnn 
        cudnnStatus_t status = cudnnCreate(&cudnn_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuDNN handle"); 
        }

        // create X and Y tensor descriptors' instances
        status = cudnnCreateTensorDescriptor(&conv_desc_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroy(cudnn_);
            throw std::runtime_error("Failed to create cuDNN conv tensor descriptor"); 
        }

        status = cudnnCreateTensorDescriptor(&bias_desc_);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(conv_desc_); 
            cudnnDestroy(cudnn_); 
            throw std::runtime_error("Failed to create cuDNN bias tensor descriptor"); 
        }


        // init X and Y tensor descriptors' instance s
        status = cudnnSetTensor4dDescriptor(conv_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            n_, c_, h_, w_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(conv_desc_); 
            cudnnDestroy(cudnn_); 
            throw std::runtime_error("Failed to init cuDNN conv tensor descriptor"); 
        }                                            

        status = cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            1, c_, 1, 1); 
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(conv_desc_); 
            cudnnDestroyTensorDescriptor(bias_desc_); 
            cudnnDestroy(cudnn_); 
            throw std::runtime_error("Failed to init cuDNN bias tensor descriptor"); 
        }                                             
    }

    ~CudnnTensorHelper() {
        cudnnDestroyTensorDescriptor(conv_desc_); 
        cudnnDestroyTensorDescriptor(bias_desc_); 
        cudnnDestroy(cudnn_); 
    }

    void addBiasToConvOutput(const std::vector<float>& conv_output, const std::vector<float> &bias, 
                                std::vector<float>&result) {
        float alpha = 1.0f; 
        float beta  = 1.0f; 
        
        std::copy(conv_output.begin(), conv_output.end(), result.begin());

        cudnnAddTensor(cudnn_, &alpha, bias_desc_, bias.data(), &beta, conv_desc_, result.data()); 
    }

private:
    cudnnHandle_t cudnn_; 
    cudnnTensorDescriptor_t conv_desc_, bias_desc_; 
    int n_; 
    int c_; 
    int h_; 
    int w_; 
}; 

// Google Test Driver class 
// helps add extra init and tear down operaiton 
class CudnnTensorTest : public ::testing::Test {
protected:
    static void SetUpTestCase() {
        // create 
        instance = new CudnnTensorHelper(1, 2, 3, 3); 
        
        // init 
        // todo 
    }

    static void TearDownTestCase() {
        // delete 
        delete instance; 

        // reset 
        instance = nullptr; 
    }

    static CudnnTensorHelper* instance; 
}; 

CudnnTensorHelper* CudnnTensorTest::instance = nullptr; 

TEST_F(CudnnTensorTest, CudnnTensorAdd) 
{
    EXPECT_NE(nullptr, instance); 

    std::vector<float> conv_output = {1,2,3,4}; 
    std::vector<float> bias = {0.5}; 
    std::vector<float> result = conv_output; 

    // execute addBiasToConvOutput 
    instance->addBiasToConvOutput(conv_output, bias, result); 

    // retrieve result and print 
    std::cout << "Result after adding bias " << std::endl; 
    for (float val : result) {
        std::cout << val << " "; 
    }
    std::cout << std::endl; 
}
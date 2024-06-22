#include <iostream>
#include <vector>
#include <stdexcept>
#include <cudnn.h>
#include <gtest/gtest.h>

TEST(TestPh1CudnnConvolutionForwardTest, ConvolutionForwardOperation) 
{
    cudnnHandle_t cudnn; 
    if ((cudnnCreate(&cudnn)) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cudnn handle"); 
    }

    // input tensor batch_size = 1, channel = 1
    // input data matrix height = 5, widt = 5 
    int n = 1, c = 1, h = 5, w = 5; 
    
    std::vector<float> input = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    // kernel tensor (1 filter, 1 input channel, height, width = 3*3)
    int kn = 1, kc = 1, kh = 3, kw; 
    std::vector<float> kernel = {
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
    };

    // output tensor 
    int out_h = h - kh + 1; 
    int out_w = w - kh + 1; 

    // expected output 
    std::vector<float> expected_output = {
        6,  6,  6,
        21, 21, 21,
        36, 36, 36
    };

    // create tensor descriptor and filter tensor descriptor 
    cudnnTensorDescriptor_t input_desc, output_desc; 

    // cudnnFilterDescriptor: cuz, hidden's filter operation is also based on the matrix
    // the filter matrix also needs descriptor to describe its batch size, channel size, height and weight
    // and filter is different from normal tensor so there created a variable of cudnn filter descriptor

    // that is to say most of the items that involves to matrix calculation 
    // in cudnn are abstracted or wraped as the objects, and objects' meta info are separated as **Descriptor_t to define
    cudnnFilterDescriptor_t kernel_desc; 
    cudnnConvolutionDescriptor_t conv_desc; 

    // create tensor descriptors and filter descriptor 
    if (cudnnCreateTensorDescriptor(&input_desc) != CUDNN_STATUS_SUCCESS) {
        cudnnDestroy(cudnn); 

        throw std::runtime_error("Failed to create input tensor descriptor"); 
    }

    if (cudnnCreateTensorDescriptor(&output_desc) != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(input_desc); 
        cudnnDestroy(cudnn); 

        throw std::runtime_error("Failed to create output tensor descriptor"); 
    }

    if (cudnnCreateFilterDescriptor(&kernel_desc) != CUDNN_STATUS_SUCCESS)  {
        cudnnDestroyTensorDescriptor(input_desc); 
        cudnnDestroyTensorDescriptor(output_desc); 
        cudnnDestroy(cudnn); 

        throw std::runtime_error("Failed to create kernel filter descriptor"); 
    }

    if (cudnnCreateConvolutionDescriptor(&conv_desc)) {
        cudnnDestroyTensorDescriptor(input_desc); 
        cudnnDestroyTensorDescriptor(output_desc); 
        cudnnDestroyFilterDescriptor(kernel_desc); 
        cudnnDestroy(cudnn); 

        throw std::runtime_error("Failed to create convolution descriptor"); 
    }

    // init descriptors(metadata)
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, kn, out_h, out_w); 
    cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, kn, kc, kh, kw); 
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); 

    // here get convolution algorithm 
    // can we specify different algorithms here ? 
    // is this the method to retrieve proper/prefer algorithm from the cudnn context via the input data and the dimensions? 
    cudnnConvolutionFwdAlgo_t algo; 
    if (cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, kernel_desc, conv_desc, output_desc,
                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,  &algo) != CUDNN_STATUS_SUCCESS) {
        // Manually set convolution algorithm
        std::cout << "Cudnn Convolution Forward Algorithm Retrieve Failed, set it by manual" << std::endl; 
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;                                    
    } 

    // here relese cudnn context resources 
    cudnnDestroyTensorDescriptor(input_desc);                                     
    cudnnDestroyTensorDescriptor(output_desc); 
    cudnnDestroyFilterDescriptor(kernel_desc);                                     
    cudnnDestroy(cudnn);  
}

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cudnn.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

// different fwd algorithms defined in the cudnn lib are in enumerations 
// this method retrieve algorithms by enumeration value 
const char* getFwdAlgoName(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return "IMPLICIT_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return "IMPLICIT_PRECOMP_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return "GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return "DIRECT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return "FFT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return "FFT_TILING";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return "WINOGRAD";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return "WINOGRAD_NONFUSED";
        default:
            return "UNKNOWN";
    }
}


TEST(TestCudnnConvolutionForwardTest, ConvolutionForwardAlgorithmSelection) 
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
    int kn = 1, kc = 1, kh = 3, kw = 3; 
    std::vector<float> kernel = {
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
    };

    // output tensor 
    int out_h = h - kh + 1; 
    int out_w = w - kh + 1; 


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
    cudnnConvolutionFwdAlgo_t cudnn_picked_best_algo;
    cudnnConvolutionFwdAlgo_t algo; 
    if (cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, kernel_desc, conv_desc, output_desc,
                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,  &algo) != CUDNN_STATUS_SUCCESS) {
        // Manually set convolution algorithm
        std::cout << "Cudnn Convolution Forward Algorithm Retrieve Failed, set it by manual" << std::endl; 
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;                                    
    } 

    // here we invoke method cudnnGetConvolutionForwardAlgorithm_v7 to retrieve the cudnn context picked prefere algorithm info 
    int returned_algo_cnt; 
    cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT]; 
    if (cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, kernel_desc, conv_desc, output_desc,
                            CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_cnt, perf_results) != CUDNN_STATUS_SUCCESS) {
        // if retrieve filed set the returned_algo_cnt value to -1 to avoid iterate 
        std::cout<< "Cunn Convolution Forward Algorithms' Performance Info Retrieve Failed, set cnt to -1" << std::endl; 
        returned_algo_cnt = -1;                                 
    }
    if (returned_algo_cnt != -1) {
        std::cout << "Available Algorithms and Correspoinding Performance Metrics " << std::endl; 
        for (int i = 0; i < returned_algo_cnt; i++) {
            std::cout << "Algorithm: " << perf_results[i].algo 
                      << ", Time: " << perf_results[i].time 
                      << ", Memory: " << perf_results[i].memory
                      << ", Status" << cudnnGetErrorString(perf_results[i].status)
                      << std::endl;                    
        }
        // here set the top best performance algorithm as the forward algorithm
        cudnn_picked_best_algo = perf_results[0].algo; 
    }

    std::cout << "System picked best cudnn_picked_best_algo name is " << getFwdAlgoName(cudnn_picked_best_algo) << std::endl; 
    std::cout << "System picked algo name is " << getFwdAlgoName(algo) << std::endl; 

    // here release cudnn context resources 
    cudnnDestroyTensorDescriptor(input_desc);                                     
    cudnnDestroyTensorDescriptor(output_desc); 
    cudnnDestroyFilterDescriptor(kernel_desc);                                     
    cudnnDestroy(cudnn);  
}

TEST(TestCudnnConvolutionForwardTest, ConvolutionForwardAlgorithmCalculation) 
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
    int kn = 1, kc = 1, kh = 3, kw = 3; 
    std::vector<float> kernel = {
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
    };

    // output tensor 
    int out_h = h - kh + 1; 
    int out_w = w - kh + 1; 
    std::vector<float> output(out_h * out_w);

    // expected output 
    std::vector<float> expected_output = {
        -6, -6, -6,
        -6, -6, -6,
        -6, -6, -6 
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

    // begin forward convolutin calculation 

    // allocate workspace 
    size_t workspace_bytes = 0; 
    void* d_workspace = nullptr; 

    if (cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, kernel_desc, 
                        conv_desc, output_desc, algo, &workspace_bytes)!= CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(input_desc);                             
        cudnnDestroyTensorDescriptor(output_desc); 
        cudnnDestroyFilterDescriptor(kernel_desc); 
        cudnnDestroyConvolutionDescriptor(conv_desc); 
        cudnnDestroy(cudnn);                             
        throw std::runtime_error("Failed to retrieve workspace bytes from context"); 
    }

    if (cudaMalloc(&d_workspace, workspace_bytes) != cudaSuccess) {                    
        throw std::runtime_error("Failed to malloc workspace bytes in cuda"); 
    }

    // here allocate device memory 
    float *d_input, *d_kernel, *d_output; 
    if (cudaMalloc(&d_input, input.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc space for d_input"); 
    }

    if (cudaMalloc(&d_output, output.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc space for d_output");
    }

    if (cudaMalloc(&d_kernel, kernel.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc space for d_kernel"); 
    }

    // copy data from host side to gpu side 
    if (cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data of input data from host to gpu space"); 
    }

    if (cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from kernel from host to gpu space"); 
    }

    // here perform convolution 
    const float alpha = 1.0f; 
    const float beta = 0.0f; 

    if (cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernel,
                                        conv_desc, algo, d_workspace, workspace_bytes, 
                                        &beta, output_desc, d_output) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to performn cudnn convolution forward operaton");                             
    }

    // copy output results from gpu to host side 
    if (cudaMemcpy(output.data() , d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data results from GPU side to Host side"); 
    }

    // here print the results 
    std::cout << "Output Data Results" << std::endl;
    for (size_t i = 0; i < output.size(); i++) {
        std::cout << output[i] << " "; 
        if ( (i + 1) % out_w == 0 ) std::cout << std::endl; 
    }

    // here compare the output and expected output value 
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_EQ(output[i], expected_output[i]); 
    }

    // here release cudnn context resources 
    cudnnDestroyTensorDescriptor(input_desc);                                     
    cudnnDestroyTensorDescriptor(output_desc); 
    cudnnDestroyFilterDescriptor(kernel_desc);                                     
    cudnnDestroy(cudnn);  
}

TEST(TestCudnnConvolutionForwardTest, ConvolutionBackwordCalculation) 
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
    int kn = 1, kc = 1, kh = 3, kw = 3; 
    std::vector<float> kernel = {
        1,  0, -1,
        1,  0, -1,
        1,  0, -1
    };

    // output tensor 
    int out_h = h - kh + 1; 
    int out_w = w - kh + 1; 
    std::vector<float> output(out_h * out_w);

    // expected output 
    std::vector<float> expected_output = {
        -6, -6, -6,
        -6, -6, -6,
        -6, -6, -6 
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

    // begin forward convolutin calculation 

    // allocate workspace 
    size_t workspace_bytes = 0; 
    void* d_workspace = nullptr; 

    if (cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, kernel_desc, 
                        conv_desc, output_desc, algo, &workspace_bytes)!= CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(input_desc);                             
        cudnnDestroyTensorDescriptor(output_desc); 
        cudnnDestroyFilterDescriptor(kernel_desc); 
        cudnnDestroyConvolutionDescriptor(conv_desc); 
        cudnnDestroy(cudnn);                             
        throw std::runtime_error("Failed to retrieve workspace bytes from context"); 
    }

    if (cudaMalloc(&d_workspace, workspace_bytes) != cudaSuccess) {                    
        throw std::runtime_error("Failed to malloc workspace bytes in cuda"); 
    }

    // here allocate device memory 
    float *d_input, *d_kernel, *d_output; 
    if (cudaMalloc(&d_input, input.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc space for d_input"); 
    }

    if (cudaMalloc(&d_output, output.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc space for d_output");
    }

    if (cudaMalloc(&d_kernel, kernel.size() * sizeof(float)) != cudaSuccess) {
        throw std::runtime_error("Failed to malloc space for d_kernel"); 
    }

    // copy data from host side to gpu side 
    if (cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data of input data from host to gpu space"); 
    }

    if (cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from kernel from host to gpu space"); 
    }

    // here perform convolution 
    const float alpha = 1.0f; 
    const float beta = 0.0f; 

    if (cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, kernel_desc, d_kernel,
                                        conv_desc, algo, d_workspace, workspace_bytes, 
                                        &beta, output_desc, d_output) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to performn cudnn convolution forward operaton");                             
    }

    // copy output results from gpu to host side 
    if (cudaMemcpy(output.data() , d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Failed to copy data results from GPU side to Host side"); 
    }

    // here print the results 
    std::cout << "Output Data Results" << std::endl;
    for (size_t i = 0; i < output.size(); i++) {
        std::cout << output[i] << " "; 
        if ( (i + 1) % out_w == 0 ) std::cout << std::endl; 
    }

    // here compare the output and expected output value 
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_EQ(output[i], expected_output[i]); 
    }
    
    float *d_grad_output, *d_grad_input; 
    cudaMalloc(&d_grad_output, output.size() * sizeof(float)); 
    cudaMalloc(&d_grad_input, input.size() * sizeof(float)); 
    cudaMemcpy(d_grad_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice); 

    cudnnConvolutionBwdDataAlgo_t bwd_data_algo; 
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn, kernel_desc, output_desc, conv_desc, input_desc, 
                               CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_data_algo); 
    size_t bwd_data_workspace_size = 0; 
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, kernel_desc, output_desc, 
                conv_desc, input_desc, bwd_data_algo, &bwd_data_workspace_size);                             
    void *d_bwd_data_workspace = nullptr;      
    cudaMalloc(&d_bwd_data_workspace, bwd_data_workspace_size);
    
    // execute convlution backward data operation 
    cudnnConvolutionBackwardData(cudnn, &alpha, kernel_desc, d_kernel, output_desc, d_grad_output, conv_desc,
                                bwd_data_algo, d_bwd_data_workspace, bwd_data_workspace_size, 
                                &beta, input_desc, d_grad_input); 
    
    cudaMemcpy(input.data(), d_grad_input, input.size() * sizeof(float), cudaMemcpyDeviceToHost);                                

    // print backward gradient generated data 
    std::cout << "Gradient Input Data" << std::endl; 
    for (int i = 0; i < h ; i++) {
        for (int j = 0; j < w; j++) {
            std::cout << input[i * w + j] << "   "; 
        }
        std::cout << std::endl; 
    }

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_workspace);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_bwd_data_workspace);
    // cudaFree(d_grad_kernel);
    // cudaFree(d_bwd_filter_workspace);
    cudnnDestroy(cudnn);                     
}
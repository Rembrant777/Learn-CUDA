#include "cudnn_activation_descriptor.h"

CudnnActivationDescriptorInstance::CudnnActivationDescriptorInstance() : 
    is_descriptor_created_(false), is_tensor_created_(false) 
{
    cudnnCreate(&cudnn_handle_); 
}

CudnnActivationDescriptorInstance::~CudnnActivationDescriptorInstance() 
{
    if (is_descriptor_created_) {
        cudnnDestroyActivationDescriptor(activation_desc_); 
    }

    if (is_tensor_created_) {
        cudnnDestroyTensorDescriptor(tensor_desc_); 
    }
    cudnnDestroy(cudnn_handle_); 
}

void CudnnActivationDescriptorInstance::createActivationDescriptor(cudnnActivationMode_t mode) 
{
    // create instance 
    if (cudnnCreateActivationDescriptor(&activation_desc_) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create activation descriptor"); 
    }

    // passing configure options 
    if (cudnnSetActivationDescriptor(activation_desc_, mode, CUDNN_PROPAGATE_NAN, 0.0) != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyActivationDescriptor(activation_desc_); 
        throw std::runtime_error("Failed to set activation descriptor"); 
    }

    // update flag 
    is_descriptor_created_ = true; 
}

cudnnActivationDescriptor_t CudnnActivationDescriptorInstance::getActivationDescriptor() const {
    return activation_desc_; 
}

// -- here after we invoke class methods to init both activaiton & tensor instance , invoke forward & backward functions here -- 


/**
 @param input: input data array pointer 
 @param output: output result data array pointer 
 @param n: batch size, batch means samples can be processed in parallel during training period. 
 @param c: channle size, means feature num 
 @param h: height, height of image 
 @param w: width, width of image 

 In FCL(Full Connection Layer also known as the Dense Layer), input features and output features 
 referring to the channel number. 

 Output = Input * Weights + Biases 
 Input      => [n, input_features]
 Weights    => [input_features, output_features]
 Biases     => [output_features]
 Output     => [n, output_features] 

*/
void CudnnActivationDescriptorInstance::computeActivationForward(float *input, float *output, 
                int n, int c, int h, int w)
{
    if (!is_tensor_created_) {
        cudnnCreateTensorDescriptor(&tensor_desc_); 
        cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 
        is_tensor_created_ = true; 
    }

    float alpha = 1.0f; 
    float beta  = 0.0f; 

    cudnnActivationForward(cudnn_handle_, activation_desc_,
            &alpha, tensor_desc_, input, &beta, tensor_desc_, output); 
}        

void CudnnActivationDescriptorInstance::computeActivationBackward(float *input, float *output, 
        float *input_diff, float *output_diff, int n, int c, int h, int w) 
{
    if (!is_tensor_created_) {
        cudnnCreateTensorDescriptor(&tensor_desc_); 
        cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w); 
        is_tensor_created_ = true; 
    }

    float alpha = 1.0f; 
    float beta  = 0.0f; 

    cudnnActivationBackward(cudnn_handle_, activation_desc_, 
            &alpha, tensor_desc_, 
            output, tensor_desc_, output_diff,
            tensor_desc_, input, &beta, 
            tensor_desc_, input_diff); 
}
#ifndef CUDNN_ACTIVATION_DESCRIPTOR_H
#define CUDNN_ACTIVATION_DESCRIPTOR_H

#include <cudnn.h>
#include <stdexcept>

class CudnnActivationDescriptorInstance {
public:
    CudnnActivationDescriptorInstance(); 
    ~CudnnActivationDescriptorInstance(); 

    void createActivationDescriptor(cudnnActivationMode_t mode); 

    cudnnActivationDescriptor_t getActivationDescriptor() const; 

    void computeActivationForward(float *input, float *output, int n, int c, int h, int w); 

    void computeActivationBackward(float *input, float *output, float *input_diff, float *output_diff, int n, int c, int h, int w);

private:
    cudnnActivationDescriptor_t activation_desc_; 
    cudnnHandle_t cudnn_handle_; 
    cudnnTensorDescriptor_t tensor_desc_; 
    bool is_descriptor_created_; 
    bool is_tensor_created_;      
}; // class  

#endif // CUDNN_ACTIVATION_DESCRIPTOR_H
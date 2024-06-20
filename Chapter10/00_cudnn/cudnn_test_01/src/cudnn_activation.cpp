#include "cudnn_activation.h"

CudnnActivation::CudnnActivation() 
{
    if (cudnnCreateActivationDescriptor(&activation_desc_) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create activation descriptor"); 
    }
}

CudnnActivation::~CudnnActivation() {
    cudnnDestroyActivationDescriptor(activation_desc_); 
}

void CudnnActivation::set_activation_descriptor(cudnnActivationMode_t mode, double coef) 
{
    if (cudnnSetActivationDescriptor(activation_desc_, mode, CUDNN_PROPAGATE_NAN, coef) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to set activation descriptor");
    }
}

void CudnnActivation::get_activation_descriptor(cudnnActivationMode_t &mode, cudnnNanPropagation_t &nanOpt, double &coef) const {
    if (cudnnGetActivationDescriptor(activation_desc_, &mode, &nanOpt, &coef) != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to get activation descriptor"); 
    }
}

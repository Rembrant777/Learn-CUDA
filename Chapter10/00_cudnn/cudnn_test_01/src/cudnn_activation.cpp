#include "cudnn_activation.h"
#include <stdexcept>

CudnnActivationInstance::CudnnActivationInstance() {
    cudnnStatus_t status = cudnnCreateActivationDescriptor(&activation_desc_);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create activation descriptor");
    }
}

CudnnActivationInstance::~CudnnActivationInstance() {
    cudnnDestroyActivationDescriptor(activation_desc_);
}

void CudnnActivationInstance::set_activation_descriptor(cudnnActivationMode_t mode, double coef) {
    cudnnStatus_t status = cudnnSetActivationDescriptor(activation_desc_, mode, CUDNN_PROPAGATE_NAN, coef);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to set activation descriptor");
    }
}

cudnnActivationDescriptor_t CudnnActivationInstance::get_activation_descriptor() const {
    return activation_desc_;
}

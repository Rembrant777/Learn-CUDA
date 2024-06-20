#ifndef CUDNN_ACTIVATION_H
#define CUDNN_ACTIVATION_H

#include <cudnn.h>
#include <stdexcept>

class CudnnActivation {
public:
    CudnnActivation(); 
    ~CudnnActivation(); 

    void set_activation_descriptor(CudnnActivationMode_t mode, double coef); 
    void get_activation_descriptor(cudnnActivationMode_t &mode, cudnnNanPropagation_t &nanOpt, double &coef) const; 

private:
    cudnnActivationDescriptor_t activation_desc_;     
    
}; // class CudnnActivation 


#endif // CUDNN_ACTIVATION_H
#ifndef CUDNN_ACTIVATION_H
#define CUDNN_ACTIVATION_H

#include <cudnn.h>
#include <stdexcept>

class CudnnActivationInstance {
public:
    CudnnActivationInstance(); 
    ~CudnnActivationInstance(); 

    void set_activation_descriptor(cudnnActivationMode_t mode, double coef); 
    cudnnActivationDescriptor_t get_activation_descriptor() const; 

private:
    cudnnActivationDescriptor_t activation_desc_;     
    
}; // class CudnnActivationInstance 


#endif // CUDNN_ACTIVATION_H
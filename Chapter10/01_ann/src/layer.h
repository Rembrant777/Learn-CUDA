#ifndef _LAYER_H_
#define _LAYER_H_

#include <string>
#include <cublas_v2.h>
#include <cudnn.h>


#include <random>

#include <cuda_runtime.h>
#include <curand.h>
#include <cassert>
#include <math.h>
#include <algorithm>

#include <sstream>
#include <fstream>
#include <iostream>


#include "blob.h"
#include "loss.h"
#include "helper.h"

namespace cudl {

class Layer {
public:
    Layer(); 
    virtual ~Layer(); 
    virtual Blob<float> *forward(Blob<float> *input) = 0; 
    virtual Blob<float> *backward(Blob<float> *grad_input)  = 0; 

    std::string get_name() {return name_; }

    virtual float get_loss(Blob<float> *target); 
    virtual int get_accuracy(Blob<float> *target); 

    void set_cuda_context(CudaContext *context) { cuda_ = context; }

    void set_load_pretrain() {
        load_pretrain_ = true; 
    }

    void set_gradient_stop() {
        gradient_stop_ = true; 
    }

    // weight freeze or un-freeze 
    void freeze() {
        freeze_ = true; 
    }

    void unfreeze() {
        freeze_ = false; 
    }
    
    // initialize weights along with the input size 
    void init_weight_bias(unsigned int seed = 1); 

    void update_weights_biases(float learning_rate); 

// I have to remote protected because my current CUDA nvcc compiler 
// do not support sub-class invoke parent defined functions, link period will raise error
// I tried to use g++ as host compiler to compile current file but it contains the cuda associated libraries so I give up to do so
// the only solution is set all functions in base class (Layer) as public 

// protected:
    virtual void fwd_initialize(Blob<float> *input) = 0; 
    virtual void bwd_initialize(Blob<float> *grad_output) = 0; 

    // name of layer 
    std::string name_; 

    // tensor descriptor for the input/output tensors 
    cudnnTensorDescriptor_t input_desc_; 
    cudnnTensorDescriptor_t output_desc_; 
    // weight/bias descriptor 
    cudnnFilterDescriptor_t filter_desc_; 
    cudnnTensorDescriptor_t bias_desc_; 

    // output memory 
    Blob<float> *input_          = nullptr; 
    Blob<float> *output_         = nullptr; 
    Blob<float> *grad_input_     = nullptr; 
    Blob<float> *grad_output_    = nullptr; 

    // master weights & bias 
    bool freeze_                = false;    // control parameter updates 
    Blob<float> *weights_       = nullptr;  // w
    Blob<float> *biases_        = nullptr;  // b 
    Blob<float> *grad_weights_  = nullptr; // dw 
    Blob<float> *grad_biases_   = nullptr; // db 

    int batch_size_ = 0; // mini-batch size 
    
    // cuda handle container 
    CudaContext *cuda_ = nullptr; 

    // pretrain parameters 
    bool load_pretrain_ = false; 
    int load_parameter(); 
    int save_parameter(); 

    // gradient stop tagging 
    bool gradient_stop_ = false; 

    friend class Network; 
}; // class Layer 

class Dense: public Layer {
public:
    Dense(std::string name, int out_size); 
    virtual ~Dense(); 

    virtual Blob<float> *forward(Blob<float> *input); 
    virtual Blob<float> *backward(Blob<float> *grad_input); 

// private:
// remove private for testing 
    void fwd_initialize(Blob<float> *input) ; 
    void bwd_initialize(Blob<float> *grad_output); 

    int input_size_     = 0; 
    int output_size_    = 0; 

    float *d_one_vec = nullptr; 
}; // class Dense 

/**
 Activation 
 1. forward: adopt the activate function upon the input data
 2. backwrd: calculate the gradient of current activate function, 
    to send loss data to previous-value. 
*/
class Activation: public Layer {
public:
    Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f); 
    virtual ~Activation(); 
    virtual Blob<float> *forward(Blob<float> *input); 
    virtual Blob<float> *backward(Blob<float> *grad_input); 

    cudnnActivationDescriptor_t get_act_desc() {
        return act_desc_; 
    }

    cudnnActivationMode_t get_act_mode() {
        return act_mode_; 
    }

    float get_coef() {
        return act_coef_; 
    }

//private:
    void fwd_initialize(Blob<float> *input); 
    void bwd_initialize(Blob<float> *output); 

    cudnnActivationDescriptor_t act_desc_; 
    cudnnActivationMode_t       act_mode_; 
    float                       act_coef_; 
}; // class Activation 


/**
Softmax includes two core parts
part-1: forward propagation
softmax_activation + loss_calculation

part-2: backward propagation 
gradient loss + weight modification 
*/
class Softmax: public Layer {
public:
    Softmax(std::string name); 
    virtual ~Softmax(); 

    virtual Blob<float> *forward(Blob<float>  *input); 
    virtual Blob<float> *backward(Blob<float> *grad_input); 
    
    float get_loss(Blob<float> *target); 
    int   get_accuracy(Blob<float> *target); 

// protected:
    void fwd_initialize(Blob<float> *input); 
    void bwd_initialize(Blob<float> *grad_output); 


    // In Softmax use CrossEntryopyLoss to calculate the loss value 
    CrossEntropyLoss loss_; 
}; // class Softmax 

 
} // namespace cudl 
#endif // _LAYER_H_
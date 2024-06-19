#ifndef _LAYER_H_
#define _LAYER_H_

#include <string>
#include <cublas_v2.h>
#include <cudnn.h>

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

protected:
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

    // initialize weights along with the input size 
    void init_weight_bias(unsigned int seed = 0);
    void update_weights_biases(float learning_rate) ; 

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

private:
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

private:
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

protected:
    void fwd_initialize(Blob<float> *input); 
    void bwd_initialize(Blob<float> *grad_output); 


    // In Softmax use CrossEntryopyLoss to calculate the loss value 
    CrossEntropyLoss loss_; 
}; // class Softmax 

 
} // namespace cudl 
#endif // _LAYER_H_
#include "network.h"
#include "helper.h"
#include "layer.h"

#include <iostream>
#include <iomanip>
#include <nvToolsExt.h>

using namespace cudl; 

Network::Network() 
{
    // do nothing 
}

Network::~Network() 
{
    // release each layer's resource and invoke layer's destroy function 
    for (auto layer : layers_) {
        delete layer; 
    }

    // release cuda context object and its resources
    if (cuda_ != nullptr) {
        delete cuda_; 
    }
}

// add layer to the network 
// new added layer will be append to the network's inner layer array 
void Network::add_layer(Layer * layer) 
{
    layers_.push_back(layer); 

    // tagging layer to stop gradient operation if it is the first layer
    if (layers_.size() == 1) {
        // disable first layer's gradient operation 
        layers_.at(0)->set_gradient_stop(); 
    }
}

// execute forward operation 
// forward operation = input data vector (dot product) weight vector + bias vector 
Blob<float> *Network::forward(Blob<float> *input) 
{
    output_ = input; 
    // here we use nvtx function to annotate the fine-grain performance report 
    nvtxRangePushA("Forward"); 
    for (auto layer : layers_) {
        #if (DEBUG_FORWARD)
        std::cout<< "[[Forward ]] [[ " << std::setw(7) << layer->get_name() << " ]]\t()" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\t"; 
        #endif // DEBUG_FORWARD

        // here invoke correspoinding specified layer {Dense, Softmax, Activation} instances to execute the 
        // acutal implementaiton of fwd_initialize and forward calculation 
        layer->fwd_initialize(output_); 

        // different Layer's sub-classes implement the fwd_initialize and forward are different 
        // depends on the different situation 
        output_ = layer->forward(output_); 

        #if (DEBUG_FORWARD)
        std::cout << "--> (" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")" << std::endl; 
        checkCudaErrors(cudaDeviceSynchronize()); 

        if (DEBUG_FORWARD > 1)
            output_->print("output", true);
            if (phase_ == inference)
                getchar(); 
        #endif 
        #endif // DEBUG_FORWARD 
    }

    nvtxRangePop(); 

    return output_; 
}

void Network::backward(Blob<float> *target)
{
    Blob<float> *gradient = target; 

    if (phase_ == inference) {
        return ; 
    }

    nvtxRangePushA("Backward"); 
    // back propagation .. update weights internally ... 
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++) {
        // getting back propagation status with gradient size 
#if (DEBUG_BACKWARD)
        std::cout << "[[Backward]] [[ " << std::setw(7) << (*layer)->get_name() << " ]] \t(" << gradient->() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\t"; 
#endif // DEBUG_BACKWARD

        (*layer)->bwd_initialize(gradient); 
        gradient = (*layer)->backward(gradient); 

#if (DEBUG_BACKWARD)
        // and the gradient result 
        std::cout << "--> (" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")" << std::endl; 
        checkCudaErrors(cudaDeviceSynchronize()); 
#endif 

#if (DEBUG_BACKWARD > 1)
        gradient->print((*layer)->get_name() + "::dx", true); 
        getchar(); 
#endif 
#endif // DEBUG_BACKWARD 
    }
    nvtxRangePop(); 
}

void Network::update(float learning_rate)
{
    if (phase_ == inference) {
        return; 
    }

#if (DEBUG_UPDATE)
    std::cout << "Start update ... lr = " << learning_rate << std::endl; 
#endif     
    nvtxRangePushA("Update"); 
    for (auto layer : layers_) {
        // if no parameters, then pass 
        if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr || 
            layer->biases_ == nullptr || layer->grad_biases_ == nullptr) {
                continue; 
        }

        layer->update_weights_biases(learning_rate); 
    }
    nvtxRangePop(); 
}

int Network::write_file() 
{
    std::cout << ".. store weights to the storage .." << std::endl; 
    for (auto layer : layers_) {
        int err = layer->save_parameter(); 

        if (err != 0) {
            std::cout << "-> error code: " << err << std::endl; 
        }

        return 0; 
    }
}

int Network::load_pretain() 
{
    for (auto layer : layers) {
        layer->set_load_pretrain(); 
    }

    return 0; 
}

// function of cuda 
// 1. initialize cuda resouce container 
// 2. register the resource container to all the layers 
void Network::cuda() 
{
    cuda_ = new CudaContext(); 
    std::cout << ".. model Configuration .. " << std::endl; 
    for (auto layer : layers_) {
        std::cout << "CUDA : " << layer->get_name() << std::endl; 
        layer->set_cuda_context(cuda_); 
    }
}

void Network::train() 
{
    phase_ = training; 
    // unfreeze all layers 
    // and why we say unfreeze here ?
    // it means each layer's parameters' {weight or bias} value can be modified and updated 
    for (auto layer: layers_) {
        layer->unfreeze(); 
    }
}

void Network::test() 
{
    phase_ = inference; 

    // freeze all layers 
    // and why we say freeze here ? 
    // it is because each layer's parameters' {weights and bias} value cannot be modified and updated 
    for (auto layer : layers_) {
        layer->freeze(); 
    }
}

std::vector<Layer*> Network::layers() 
{
    return layers_; 
}

float Network::loss(Blob<float> *target)
{
    Layer *layer = layers_.back(); 
    return layer->get_loss(target); 
}

int Network::get_accuracy(Blob<float> *target) 
{
    Layer *layer = layers_.back(); 
    return layer->get_accuracy(target); 
}

#if 0
Blob<float> *predict = this->output_; 
    int batch_size = predict->n(); 
    int output_size = precision->c(); 

#if (DEBUG_ACCRACY)
    std::cout << "[[ ACCURACY ]]" << std::endl; 
    predict->print("predict:", true); 
    target->print("target", true); 
#endif // DEBUG_ACCURACY 

    float* h_predict = predict->to(host); 
    float* h_target = target->to(host); 
    cudaDeviceSynchronize(); 
    int result = 0; 
    for (int b = 0; b < batch_size; b++) {
        int idx_predict = 0; 
        int idx_target = 0; 
        for (int j = 0; j < output_size; j++) {
            if (h_predict[b * output_size + j] > h_predict[idx_predict]) {
                idx_predict = j; 
            }
            if (h_target[b * output_size + j] > h_target[idx_target]) {
                idx_target = j; 
            }
        }
#if (DEBUG_ACCRACY)        
        std::cout << "predict:: " << idx_predict << ", target::" << idx_target << std::endl; 
#endif  // DEBUG_ACCURACY 
        
        if (idx_predict == idx_target) {
            result++; 
        }
    }
    
#if (DEBUG_ACCRACY)    
    getchar(); 
#endif // DEBUG_ACCRUACY 
#endif     
#include "layer.h"
#include <random>
#include <cuda_runtime.h>
#include <curand.h>
#include <cassert>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace cudl; 

// Layer Definition 

Layer::Layer() {
    // do nothing 
}

Layer::~Layer() {
#if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0) 
    std::cout << "Destroy Layer: " << name_ << std::endl; 
#endif     
	if (output_       != nullptr) { delete output_;       output_       = nullptr; }
	if (grad_input_   != nullptr) { delete grad_input_;   grad_input_   = nullptr; }

	if (weights_      != nullptr) { delete weights_;      weights_	    = nullptr; }
	if (biases_       != nullptr) { delete biases_;	      biases_       = nullptr; }
	if (grad_weights_ != nullptr) { delete grad_weights_; grad_weights_ = nullptr; }
	if (grad_biases_  != nullptr) { delete grad_biases_;  grad_biases_  = nullptr; }
}

void Layer::init_weight_bias(unsigned int seed)
{
    checkCudaErrors(cudaDeviceSynchronize()); 

    if (weights_ == nullptr || biases_ == nullptr) {
        return; 
    }

    // Create random network 
    std::random_device rd; 

    // initialize a Mersenne Twister 
    // if recv seed value is 0, use rd() return value as seed value to Mersenne Twister 
    // otherwise use received seed value 
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    // He uniform distribution 
    // He's initialization 
    float range = sqrt(6.f / input_->size()); 
    std::uniform_real_distribution<> dis(-range, range); 

    for (int i = 0; i < weights_->len(); i++) {
        weights_->ptr()[i] = static_cast<float>(dis(gen)); 
    }

    for (int i = 0; i < biases_->len(); i++) {
        biases_->ptr()[i] = 0.f; 
    }

    // copy initialized value to the device 
    weights_->to(DeviceType::cuda); 
    biases_->to(DeviceType::cuda); 

    std::cout << ".. initialized " << name_ << " layer .." << std::endl; 
}

void Layer::update_weights_biases(float learning_rate) 
{
    float eps = -1.f * learning_rate; 
    if (weights_ != nullptr && grad_weights_ != nullptr) {
#if (DEBUG_UPDATE)    
        weights_->print(name_ + "::weights (before update)", true); 
        grad_weights_->print(name_ + "::gweights", true); 
#endif // DEBUG_UPDATE

        // w(new) = w(old) + eps * dw
        checkCublasErrors(
            cublasSaxpy(
                cuda_->cublas(),
                &eps,
                grad_weights_->cuda(), 1,
                weights_->cuda(), 1));

#if (DEBUG_UPDATE)                 
        weights_->print(name_ + "weights (after update)", true); 
#endif // DEBUGUPDATE
    }

    if (biases_ != nullptr && grad_biases_ != nullptr) {
#if (DEBUG_UPDATE)
        biases_->print(name_ + "biases (before update)", true); 
        grad_biases_->print(name_ + "gbiases", true); 
#endif // DEBUG_UPDATE 

        // b = b + eps * db 
        checkCublasErrors(
            cublasSaxpy(cuda_->cublas(),
                biases_->len(),
                &eps,
                grad_biases_->cuda(), 1,
                biases_->cuda(), 1)); 
#if (DEBUG_UPDATE)
        biases_->print(name_ + "biases (after update)", true); 
#endif 
    }
}

float Layer::get_loss(Blob<float> *target)
{
    assert("No Loss layer has no loss." && false);
    return EXIT_FAILURE;  
}

int Layer::get_accuracy(Blob<float> *target)
{
    assert("No Loss layer cannot estimate accuracy." && false); 
    return EXIT_FAILURE; 
}

int Layer::load_parameter() 
{

}

int Layer::save_parameter() 
{

}
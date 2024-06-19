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
    std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

    // He 

}
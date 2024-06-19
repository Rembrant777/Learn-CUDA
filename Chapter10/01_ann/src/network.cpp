#include "network.h"
#include "helper.h"
#include "layer.h"

#include <iostream>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

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

        // here invoke correspoinding specified layer {Dense, Softmax, Activation} instances to execute the 
        // acutal implementaiton of fwd_initialize and forward calculation 
        layer->fwd_initialize(output_); 

        // different Layer's sub-classes implement the fwd_initialize and forward are different 
        // depends on the different situation 
        output_ = layer->forward(output_); 
    }
}
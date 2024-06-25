#include "../src/loss.h"
#include "../src/layer.h"
#include "../src/helper.h"
#include <gtest/gtest.h>

using namespace cudl; 
using namespace std;

TEST(TestLayer, DenseLayerCreateTest) {
    string name = "test_dense_layer_1"; 
    int output_size = 10; 
    Dense* layer = new Dense(name, output_size); 
    EXPECT_NE(layer, nullptr); 
    EXPECT_NE(layer, nullptr); 

    delete layer; 
}

TEST(TestLayer, ActivationLayerCreateTest) {
    string name = "test_activation_layer_1"; 
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU; 
    Activation* layer = new Activation(name, mode); 
    EXPECT_NE(layer, nullptr); 

    delete layer; 
}

TEST(TestLayer, SoftmaxLayerCreateTest) {
    string name = "softmax-layer-1"; 
    Softmax* layer = new Softmax(name); 
    EXPECT_NE(layer, nullptr); 

    delete layer; 
}

/**
 In this test, we test 
 0. create Dense Layer and 
 1. init weight & bias 
 2. write parameters(weight & bias) to local file 
 3. load parameters(weight & bias) from local file and verify 
*/
TEST(Testlayer, DenseInitWeightBias) {
    string d_name = "dense-layer-0"; 
    int output_size = 8; 
    Dense* d_layer = new Dense(d_name, output_size); 
    EXPECT_NE(d_layer, nullptr);

    int n = 1, c = 2, h = 3, w = 4; 
    Blob<float>* input = new Blob<float>(n, c, h, w); 
    d_layer->unfreeze(); 
    d_layer->fwd_initialize(input); 
    

    // init biases cannot be null 
    Blob<float>* p_biases = d_layer->biases_; 
    EXPECT_NE(p_biases, nullptr); 

    // init weights cannot be null
    Blob<float>* p_weights = d_layer->weights_; 
    EXPECT_NE(p_weights, nullptr); 

    // init input_ cannot be null 
    // input's tensor (input data dimension meta info object) neither cannot be null 
    EXPECT_EQ(input, d_layer->input_);
    EXPECT_NE(nullptr, d_layer->input_->tensor()); 

    // init output_ cannot be null 
    EXPECT_NE(nullptr, d_layer->output_);
    // output's tensor descriptor neither cannot be null
    EXPECT_NE(nullptr, d_layer->output_->tensor()); 

    // init forward will create & init d_one_vec 
    EXPECT_NE(nullptr, d_layer->d_one_vec); 

    // weight & bias init value are initialized in fwd_initialize 
    // d_layer->init_weight_bias(); 
    p_biases->print("bias-value", 1, 4); 
    p_weights->print("weight-value", 1, 4); 


    // here we check class variable load_pretrain_ 
    // if load_pretrain_ = true it will load parameters(weight & bias) from file 
    // otherwise dense layer will invoke init_weights_biases function to generate parameters and hold them in memory 
    EXPECT_EQ(false, d_layer->load_pretrain_); 

    // since every thing is ok, we mock the network layer's write its parameters to disk
    // then let next layer to load parameter from the disk 
    //int ret = d_layer->save_parameter(); 
    // if parameters are write success, ret value should be 
    // EXPECT_EQ(0, ret); 


    delete d_layer; 
}
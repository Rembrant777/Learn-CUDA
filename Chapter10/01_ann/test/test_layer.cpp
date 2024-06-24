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
    

    Blob<float>* p_biases = d_layer->biases_; 
    EXPECT_NE(p_biases, nullptr); 
    Blob<float>* p_weights = d_layer->weights_; 
    EXPECT_NE(p_weights, nullptr); 

    d_layer->init_weight_bias(); 

    delete d_layer; 
}
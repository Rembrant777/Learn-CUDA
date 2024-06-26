#include "../src/network.h"
#include <gtest/gtest.h>

using namespace std; 
using namespace cudl; 

TEST(TestNetwork, TestCreateAndInitNetwork) {
    Network *network = new Network(); 
    EXPECT_NE(network, nullptr);

    // cuda should be null 
    EXPECT_EQ(network->get_cuda_context(), nullptr);

    network->cuda(); 
    // cuda cannot be null 
    EXPECT_NE(network->get_cuda_context(), nullptr); 

    // network's layer should be null without adding any layer instances 
    vector<Layer*> layer_list = network->layers(); 
    EXPECT_TRUE(layer_list.empty()); 

    EXPECT_EQ(network->get_network_workload(), inference);

    delete network; 
}

TEST(TestNetwork, TestSingleDenseLayerNetwork) {
    Network* network = new Network(); 
    EXPECT_NE(network, nullptr); 

    // creat input/output/grad_output/grad_input data's batch, channel, weight value
    int n = 3, c = 4, h = 9, w = 10; 

    string name = "dense-layer"; 
    // create Dense Layer 
    Dense* layer = new Dense(name, w); 

    // init network's cuda context
    network->cuda(); 

    EXPECT_NE(nullptr, network->get_cuda_context()); 

    // add layer to network 
    network->add_layer(layer); 

    // get layer from  network and check its inner variables 
    Dense* layer_net = (Dense *)network->layers().at(0); 
    EXPECT_NE(layer_net, nullptr);

    // inner layer's gradient stop flag should be updated to true 
    EXPECT_EQ(true, layer_net->get_gradient_stop()); 

    // we do not want layer load parameters(w, b) from local file 
    // so we disable this option via set(or make sure) that 
    // layer#freeze = false and load_pretrain = false 
    EXPECT_EQ(false, layer_net->get_freeze()); 
    EXPECT_EQ(false, layer_net->get_load_pretrain()); 


    // then we create input data instance and init its data 
    Blob<float>* input = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, input); 

    input->gen_mock_data_for_predict();  
    input->print_data("network-layer-input-data", input->n(), input->w()); 

    network->cuda(); 

    // make sure network's inner each layer context of cuda is available 
    EXPECT_NE(nullptr, layer_net->cuda_); 

    // then we invoke the forward calculation via network 
    Blob<float>* output = network->forward(input); 
    EXPECT_NE(nullptr, output); 
    output->print_data("network-layer-output-data", output->n(), output->w());

    // here we continue with the backward calculation and print results
    Blob<float>* grad_output = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, grad_output);

    Blob<float>* grad_input = layer_net->grad_input_; 
    // grad_input value should not be initialized 
    grad_input->print_data("grad input value before backward calculation", 
                            grad_input->n(), grad_input->w()); 

    // gen mock dataset
    grad_output->gen_mock_data_for_predict(); 
    network->backward(grad_output); 
    EXPECT_NE(nullptr, grad_input); 

    // grad_input value should be updated after network invoke backward  
    grad_input->print_data("grad input value before backward calculation", 
                            grad_input->n(), grad_input->w()); 

    delete network; 
}

TEST(TestNetwork, TestSingleActivationLayerNetwork) {
    Network* network = new Network(); 
    EXPECT_NE(network, nullptr); 
    int n = 3, c = 4, h = 9, w = 10; 
    string name = "activation-layer"; 
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU; 
    float coef = 0.0f; 

    Activation* layer = new Activation(name, mode, coef); 
    EXPECT_NE(nullptr, layer); 

    network->add_layer(layer); 

    Blob<float>* input = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, input); 

    input->gen_mock_data_for_predict(); 
    input->print_data("network-act-layer-input", input->n(), input->w()); 

    network->cuda(); 
    
    EXPECT_NE(nullptr, network->get_cuda_context()); 
    EXPECT_NE(nullptr, network->layers().at(0)->cuda_); 

    Blob<float>* output = network->forward(input);
    EXPECT_NE(output, nullptr); 
    output->print_data("network-act-layer-output", output->n(), output->w()); 

    // here we continue with the backward calculation and print results
    Blob<float>* grad_output = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, grad_output);

    // gen mock dataset
    grad_output->gen_mock_data_for_predict(); 
    network->backward(grad_output); 

    Blob<float>* grad_input = network->layers().at(0)->grad_input_; 
         EXPECT_NE(nullptr, grad_input); 

    // grad_input value should not be initialized 
    grad_input->print_data("grad input value before backward calculation", 
                            grad_input->n(), grad_input->w()); 

    delete network; 
}

TEST(TestNetwork, TestSingleSoftmaxLayerNetwork) {
    Network* network = new Network(); 
    EXPECT_NE(network, nullptr); 
    int n = 3, c = 4, h = 9, w = 10; 
    string name = "softmax-layer"; 

    Softmax* layer = new Softmax(name); 
    EXPECT_NE(nullptr, layer); 

    network->add_layer(layer); 

    Blob<float>* input = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, input); 

    input->gen_mock_data_for_predict(); 
    input->print_data("network-softmax-layer-input", input->n(), input->w()); 

    network->cuda(); 
    
    EXPECT_NE(nullptr, network->get_cuda_context()); 
    EXPECT_NE(nullptr, network->layers().at(0)->cuda_); 

    Blob<float>* output = network->forward(input);
    EXPECT_NE(output, nullptr); 
    output->print_data("network-softmax-layer-output", output->n(), output->w()); 

    Blob<float>* grad_output = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, grad_output);
    // gen mock dataset
    grad_output->gen_mock_data_for_predict(); 
    network->backward(grad_output); 

    Blob<float>* grad_input = network->layers().at(0)->grad_input_; 

    EXPECT_NE(nullptr, grad_input); 
    // grad_input value should be updated after network invoke backward  
    grad_input->print_data("grad input value before backward calculation", 
                            grad_input->n(), grad_input->w()); 

    delete network; 
}

TEST(TestNetwork, TestCreateMultipleMixLayerNetwork) {
    Network *network = new Network(); 
    network->add_layer(new Dense("dense1", 500)); 
    network->add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU)); 
    network->add_layer(new Dense("dense2", 10)); 
    network->add_layer(new Softmax("softmax")); 
    network->cuda(); 

    EXPECT_NE(network, nullptr); 
    EXPECT_EQ(network->layers().size(), 4);
    EXPECT_NE(network->layers().at(0), nullptr);
    EXPECT_NE(network->layers().at(1), nullptr);
    EXPECT_NE(network->layers().at(2), nullptr);
    EXPECT_NE(network->layers().at(3), nullptr);

    delete network; 
}

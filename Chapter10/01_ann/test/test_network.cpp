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

TEST(TestNetwork, TestCreateSingleDenseLayerNetwork) {
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
    // output->print_data("network-layer-output-data", output->n(), output->w());

    delete network; 
}

TEST(TestNetwork, TestCreateSingleActivationLayerNetwork) {
    EXPECT_EQ(1, 1);
}

TEST(TestNetwork, TestCreateSingleSoftmaxLayerNetwork) {
    EXPECT_EQ(1, 1);
}

TEST(TestNetwork, TestCreateMultipleMixLayerNetwork) {
    EXPECT_EQ(1, 1);
}

TEST(TestNetwork, TestCreateNetworkIterateSingleTime) {
    EXPECT_EQ(1, 1); 
}
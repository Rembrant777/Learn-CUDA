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
    EXPECT_EQ(1, 1);
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
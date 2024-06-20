// in this test case file we use cudnnActivationDescriptor_t 
// to create multiple types activation function support instances. 
// and use the instances of cudnnActivationDescriptor_t to invoke correspoinding 
// 1. cudnnActivationBackward
// 2. cudnnActivationForward functions and retrieve results 
#include <gtest/gtest.h>
#include "cudnn_activation_descriptor.h"

class CudnnActivationTest : public ::testing::Test {
protected:
    CudnnActivationTest(): activation() {}
    CudnnActivationDescriptorInstance activationDescriptorInstance;     
}; 

TEST_F(CudnnActivationDescriptorTest, CreateAndDestroy) {
    

}
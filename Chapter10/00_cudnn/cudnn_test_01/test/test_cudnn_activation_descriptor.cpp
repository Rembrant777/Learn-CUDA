// in this test case file we use cudnnActivationDescriptor_t 
// to create multiple types activation function support instances. 
// and use the instances of cudnnActivationDescriptor_t to invoke correspoinding 
// 1. cudnnActivationBackward
// 2. cudnnActivationForward functions and retrieve results 
#include <gtest/gtest.h>
#include "cudnn_activation_descriptor.h"

class CudnnActivationDescriptorTest {
protected:
    CudnnActivationDescriptorTest(): activation() {}
    CudnnActivationDescriptorInstance activationDescriptorInstance;     
    const int n = 1, c = 1, h = 2, w = 2; 

    std::vector<float> input = {1.0, 2.0, 3.0, 4.0}; 
    std::vector<float> output = {0.0, 0.0, 0.0, 0.0}; 
    std::vector<float> input_diff = {0.1, 0.2, 0.3, 0.4};
    std::vector<float> output_diff = {0.0, 0.0, 0.0, 0.0};
}; 

// activation function -> sigmoid 
TEST(CudnnActivationDescriptorTest, CreateSigmoidActivationDescriptor) {
    // invoke method to create && init activation descriptor with activate function as sigmoid 
    EXPECTE_NO_THROW(activation.CreateSigmoidActivationDescriptor()); 

    // retrieve instance of activaiton descriptor via get method 
    cudnnActivationDescriptor_t desc = activation.getActivationDescriptor(); 

    // execute assertion 
    EXPECT_NE(desc, nullptr); 

    cudnnActivationMode_t mode; 
    cudnnNanPropagation_t nan_prop; 
    double coef; 

    // here retrieve configure options from the cudnn activation descriptor instance 
    // mode     <- get_from_cudnn_activation(instance of activation descriptor)
    // nan_prop <- get_from_cudnn_activation(instance of activation descriptor)
    // coef     <- get_from_cudnn_activation(instance of activation descriptor) 
    cudnnGetActivationDescriptor(desc, &mode, &nan_prop, &coef); 

    // execute the equation validation via the retrieved configure option values 
    // nan_prop configure option determines how to handle NAN value 
    // coef configure option means the coefficient value, Sigmoid's cofficient does not work 
    EXPECT_EQ(mode, CUDNN_ACTIVATION_SIGMOD); 
    EXPECT_EQ(nan_prop, CUDNN_PROPAGATE_NAN); 

}
#include <gtest/gtest.h>
#include "cudnn_activation.h"

class CudnnActivationTest : public ::testing::Test {
protected:
    CudnnActivationTest(): activation() {}
    CudnnActivation activation;     
}; 

TEST_F(CudnnActivationTest, CreateAndDestroy) {

}

TEST_F(CudnnActivationTest, SetAndGetActivationDescriptor) {
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU; 
    double coef = 0.0; 

    activation.set_activation_descriptor(mode, coef); 

    cudnnActivationMode_t returned_mode; 
    cudnnPropagation_t    returned_nan_opt; 
    double returned_coef; 

    activation.get_activation_descriptor(returned_mode, returned_nan_opt, returned_coef); 

    EXPECT_EQ(returned_mode, mode); 
    EXPECT_EQ(returned_nan_opt, CUDNN_PROPAGATE_NAN); 
    EXPECT_EQ(returned_coef, coef); 
}

TEST_F(CudnnActivationTest, SetInvalidActivationDescriptor) 
{
    cudnnActivationMode_t mode = static_cast<cudnnActivationMode_t>(-1); 
    doule coef = 0.0; 

    EXPECT_THROW(activation.set_activation_descriptor(mode, coef), std::runtime_error); 
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv); 
    return RUN_ALL_TESTS(); 
}


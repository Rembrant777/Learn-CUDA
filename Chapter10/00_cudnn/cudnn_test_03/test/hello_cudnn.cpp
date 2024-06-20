#include <gtest/gtest.h>
#include <cudnn.h>
#include <stdexcept>

TEST(HelloCudnn, CreateActivationDescriptorSigmoid) {
    // assert create Sigmoid Activation Descriptor Success 
    // cudnn handler 
    cudnnHandle_t cudnn_handle; 

    // create cunn context 
    cudnnCreate(&cudnn_handle); 

    cudnnActivationMode_t origin_mode = CUDNN_ACTIVATION_SIGMOID; 
    EXPECT_EQ(origin_mode, CUDNN_ACTIVATION_SIGMOID); 

    // create instance of cudnn activate descriptor 
    cudnnActivationDescriptor_t activation_desc; 
    cudnnCreateActivationDescriptor(&activation_desc); 

    // init cudnn activate descriptor 
    // set coefficient value which does not work in sigmoid type activate function 
    float origin_coef = 0.0f; 
    cudnnNanPropagation_t origin_nan_prop = CUDNN_PROPAGATE_NAN; 
    cudnnSetActivationDescriptor(activation_desc, origin_mode, origin_nan_prop, origin_coef);

    // -- here retrieve mode, nanOp and coef values in the instance of cudnn activation function via cudnn context 
    cudnnActivationMode_t ret_mode;
    cudnnNanPropagation_t ret_nan_prop;
    double ret_coef;
    cudnnGetActivationDescriptor(activation_desc, &ret_mode, &ret_nan_prop, &ret_coef); 

    // here execute eq validation 
    EXPECT_EQ(ret_coef, origin_coef); 
    EXPECT_EQ(ret_mode, origin_mode); 
    EXPECT_EQ(ret_nan_prop, origin_nan_prop); 

    // here execute destroy cudnn resource operaitons 
    cudnnDestroyActivationDescriptor(activation_desc); 
    cudnnDestroy(cudnn_handle);
}
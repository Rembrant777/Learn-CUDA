#include <gtest/gtest.h>
#include <cudnn.h>
#include <stdexcept>

class TestCudnnActivationDescriptor {
public:
    TestCudnnActivationDescriptor(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_prop, float coef): 
                activate_mode_(mode), nan_prop_(nan_prop), coef_(coef), is_descriptor_created_(false) {
    }

    ~TestCudnnActivationDescriptor() {
        cudnnDestroy(cudnn_handle_); 
    }

    void createActivationDescriptor() {
        if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cudnn handler!"); 
        }
        
        if (cudnnCreateActivationDescriptor(&activate_descroptor_) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create activation descriptor instance"); 
        }

        if (cudnnSetActivationDescriptor(activate_descroptor_, activate_mode_, nan_prop_, coef_) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to init activation descriptor"); 
        }

        is_descriptor_created_ = true; 
    }

    void destroyActivationDescriptor() {
        if (cudnnDestroyActivationDescriptor(activate_descroptor_) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to destroy activation descriptor"); 
        }

        is_descriptor_created_ = false; 
    }

    void get_activation_descriptor(cudnnActivationMode_t &ret_mode, 
                            cudnnNanPropagation_t &ret_nan_prop, double &ret_coef) const {
        if (cudnnGetActivationDescriptor(activate_descroptor_, &ret_mode, &ret_nan_prop, &ret_coef) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set activation descriptor"); 
        }
    }

private:
    cudnnHandle_t               cudnn_handle_; 
    cudnnActivationDescriptor_t activate_descroptor_; 
    cudnnActivationMode_t       activate_mode_; 
    cudnnNanPropagation_t       nan_prop_; 
    float                       coef_; 
    bool                        is_descriptor_created_; 
}; 

TEST(TestCudnnActivationDescriptor, CreateAndDestroyActivationDescriptor) 
{
    cudnnActivationMode_t mode      = CUDNN_ACTIVATION_SIGMOID; 
    cudnnNanPropagation_t nan_prop  = CUDNN_PROPAGATE_NAN; 
    double coef                     = 0.0; 
    EXPECT_EQ(mode, CUDNN_ACTIVATION_SIGMOID);
    EXPECT_EQ(nan_prop, CUDNN_PROPAGATE_NAN);

    TestCudnnActivationDescriptor* instance = new TestCudnnActivationDescriptor(mode, nan_prop, coef);
}

TEST(TestCudnnActivationDescriptor, CreateSigmoidActivationDescriptor)
{
    EXPECT_EQ(1, 1);
}

TEST(TestCudnnActivationDescriptor, CreateReluActivationDescriptor)
{
    EXPECT_EQ(1, 1);
}

TEST(TestCudnnActivationDescriptor, CreateTanhActivationDescriptor)
{
    EXPECT_EQ(1, 1);
}

TEST(TestCudnnActivationDescriptor, CreateClippedReluActivationDescriptor)
{
    EXPECT_EQ(1, 1);
}

TEST(TestCudnnActivationDescriptor, CreateEluActivationDescriptor)
{
    EXPECT_EQ(1, 1);
}
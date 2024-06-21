#include <gtest/gtest.h>
#include <cudnn.h>
#include <stdexcept>
// @Deprecated not gonna implement this unit test case 
class TestCudnnActivationDescriptor {
public:
    TestCudnnActivationDescriptor(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_prop, float coef): 
                activate_mode_(mode), nan_prop_(nan_prop), coef_(coef), is_descriptor_created_(false) {
    }

    ~TestCudnnActivationDescriptor() {
        if (is_descriptor_created_) {
            cudnnDestroy(cudnn_handle_); 
        }
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

    // method to retrieve Activation Descriptor Options from Cudnn Context Env Configure Options. 
    void getActivationDescriptor(cudnnActivationMode_t &ret_mode, 
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

// enumeration: CUDNN_ACTIVATION_ELU
TEST(TestCudnnActivationDescriptor, CreateEluActivationDescriptor)
{
    cudnnActivationMode_t mode      = CUDNN_ACTIVATION_ELU; 
    cudnnNanPropagation_t nan_prop  = CUDNN_PROPAGATE_NAN; 
    double coef                     = 0.0; 
    EXPECT_EQ(mode, CUDNN_ACTIVATION_ELU);
    EXPECT_EQ(nan_prop, CUDNN_PROPAGATE_NAN);

    TestCudnnActivationDescriptor* instance = new TestCudnnActivationDescriptor(mode, nan_prop, coef);
    EXPECT_NE(instance, nullptr); 

    EXPECT_NO_THROW((*instance).createActivationDescriptor()); 

    cudnnActivationMode_t ret_mode; 
    cudnnNanPropagation_t ret_nan_prop; 
    double ret_coef; 

    EXPECT_NO_THROW((*instance).getActivationDescriptor(ret_mode, ret_nan_prop, ret_coef)); 

    EXPECT_EQ(ret_mode, mode); 
    EXPECT_EQ(ret_nan_prop, nan_prop); 
    EXPECT_EQ(ret_coef, coef); 

    EXPECT_NO_THROW((*instance).destroyActivationDescriptor()); 
}


TEST(TestCudnnActivationDescriptor, ApplyEluActivatioForward)
{
    EXPECT_EQ(1, 1);
}


TEST(TestCudnnActivationDescriptor, ApplyEluActivatioBackforward)
{
    EXPECT_EQ(1, 1);
}
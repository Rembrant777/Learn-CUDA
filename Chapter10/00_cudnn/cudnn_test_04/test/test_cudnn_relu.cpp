#include <gtest/gtest.h>
#include <cudnn.h>
#include <stdexcept>

class CudnnReluActivationDescriptor {
public:
    CudnnReluActivationDescriptor(cudnnActivationMode_t mode, cudnnNanPropagation_t nan_prop, float coef): 
                activate_mode_(mode), nan_prop_(nan_prop), coef_(coef), is_descriptor_created_(false) {
    }

    ~CudnnReluActivationDescriptor() {
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
        std::cout << "#destroyActivationDescriptor " << std::endl; 
        if (cudnnDestroyActivationDescriptor(activate_descroptor_) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to destroy activation descriptor"); 
        }

        is_descriptor_created_ = false; 
    }

    // method to retrieve Activation Descriptor Options from Cudnn Context Env Configure Options. 
    void getActivationDescriptor(cudnnActivationMode_t &ret_mode, 
                            cudnnNanPropagation_t &ret_nan_prop, double &ret_coef) const {

        // if  not created return without set values                                 
        if (!is_descriptor_created_) {
            return; 
        }                                            

        if (cudnnGetActivationDescriptor(activate_descroptor_, &ret_mode, &ret_nan_prop, &ret_coef) != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to set activation descriptor"); 
        }
    }
    // here we implement MSE as the loss function 
    // MSE (Mean Sequared Error)
    // MSE = { ((y(1) - y'(1))^2 + ... + (y(n) - y'(n))^2 } /n, n is the number of sample 
    double mseLossFunc(const float *predicted, const float* expected, int N) {
        double ret = 0.0; 
        for (int i = 0; i < N; i++) {
            float diff = predicted[i] - expected[i]; 
            ret += diff * diff;  
            
            // std::cout << "diff " << diff << ", ret " << ret << std::endl; 
        }

        // std::cout << "ret " << ret << ", N " << N << std::endl; 
        // std::cout << "ret / N = " << (ret / N) << std::endl; 
        ret /= N; 
        return ret; 
    }

    // here we implement MES's gradient function 
    // MSE' = 2 * { ((y(1) - y'(1)) + ... + (y(n) - y'(n)) } / n
    // MSE' = {2 * ((y(1) - y'(1)) + ... + 2 * ((y(n) - y'(n)) } / n 
    // retrieve class inner activation descriptor instance 
    void mseLossGradient(const float *predicted, const float *expected, float *grad, int N) {
        for (int i = 0; i < N; i++) {
            grad[i] = 2 * (predicted[i] - expected[i]) / N; 
        }
    }


    float huberLoss(float *predicted, float *expected, int delta, int N) {
        float ret = 0.f; 
        for (int i = 0; i < N; i++) {
            float diff = predicted[i] - expected[i]; 
            if (abs(diff) <= delta) {
                diff = 0.5 * diff * diff; 
            } else {
                diff = delta * (abs(diff) - 0.5 * delta); 
            }

            ret += diff; 
        }

        return ret / N; 
    }

    void huberLossDerivative(float *predicted, float *expected, float *grad_output, float delta, int N) {
        for (int i = 0; i < N; i++) {
            float diff = predicted[i] - expected[i]; 
            if (abs(diff) <= delta) {
                grad_output[i] = diff; 
            } else {
                grad_output[i] = delta * (diff > 0 ? 1 : -1); 
            }
        }
    }

    cudnnActivationDescriptor_t getActivationDescriptor() {
        return activate_descroptor_; 
    }

    cudnnHandle_t getCudnnHandle() {
        return cudnn_handle_; 
    }

private:
    cudnnHandle_t               cudnn_handle_; 
    cudnnActivationDescriptor_t activate_descroptor_; 
    cudnnActivationMode_t       activate_mode_; 
    cudnnNanPropagation_t       nan_prop_; 
    float                       coef_; 
    bool                        is_descriptor_created_; 
}; 

class CudnnActivationTest : public ::testing::Test {
protected:
    static void SetUpTestCase() {
        instance = new CudnnReluActivationDescriptor(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
        instance->createActivationDescriptor();
    }

    static void TearDownTestCase() {
        delete instance;
        instance = nullptr;
    }

    static CudnnReluActivationDescriptor* instance;
};

CudnnReluActivationDescriptor* CudnnActivationTest::instance = nullptr;

// enumeration: CUDNN_ACTIVATION_RELU
TEST_F(CudnnActivationTest, CreateReluActivationDescriptor) 
{
    cudnnActivationMode_t mode      = CUDNN_ACTIVATION_RELU; 
    cudnnNanPropagation_t nan_prop  = CUDNN_PROPAGATE_NAN; 
    double coef                     = 0.0; 

    EXPECT_NE(instance, nullptr);

    cudnnActivationMode_t ret_mode; 
    cudnnNanPropagation_t ret_nan_prop; 
    double ret_coef; 

    EXPECT_NO_THROW((*instance).getActivationDescriptor(ret_mode, ret_nan_prop, ret_coef)); 

    EXPECT_EQ(ret_mode, mode); 
    EXPECT_EQ(ret_nan_prop, nan_prop); 
    EXPECT_EQ(ret_coef, coef);
}

TEST_F(CudnnActivationTest, ApplyReluActivationForward)
{
    EXPECT_EQ(1, 1);
}


TEST_F(CudnnActivationTest, ApplyReluActivationBackforward)
{
    EXPECT_EQ(1, 1);
}
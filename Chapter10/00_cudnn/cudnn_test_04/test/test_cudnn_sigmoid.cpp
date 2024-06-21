#include <gtest/gtest.h>
#include <cudnn.h>
#include <cmath>
#include <stdexcept>

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
        instance = new TestCudnnActivationDescriptor(CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0);
        instance->createActivationDescriptor();
    }

    static void TearDownTestCase() {
        delete instance;
        instance = nullptr;
    }

    static TestCudnnActivationDescriptor* instance;
};

TestCudnnActivationDescriptor* CudnnActivationTest::instance = nullptr;


// enumeration: CUDNN_ACTIVATION_SIGMOID
TEST_F(CudnnActivationTest, CreateSigmoidActivationDescriptor) 
{
    cudnnActivationMode_t mode      = CUDNN_ACTIVATION_SIGMOID; 
    cudnnNanPropagation_t nan_prop  = CUDNN_PROPAGATE_NAN; 
    double coef                     = 0.0; 
    EXPECT_EQ(mode, CUDNN_ACTIVATION_SIGMOID);
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
}

TEST_F(CudnnActivationTest, ApplySigmoidActivationForward)
{
    cudnnActivationMode_t mode      = CUDNN_ACTIVATION_SIGMOID; 
    cudnnNanPropagation_t nan_prop  = CUDNN_PROPAGATE_NAN; 
    double coef                     = 0.0; 
    cudnnActivationDescriptor_t act_desc = (*instance).getActivationDescriptor(); 


    const int batch_size = 2; 
    const int channels   = 3; 
    const int height     = 4; 
    const int width      = 4; 
    const int size       = batch_size * channels * height * width ; 

    float input[size]   =  {
                                1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 2.1, -2.1, 0.2, -0.2, 2.2, -2.2,
                                1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 2.1, -2.1, 0.2, -0.2, 2.2, -2.2,
                                1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 2.1, -2.1, 0.2, -0.2, 2.2, -2.2
                            };

    float output[size]; 

    // create & cudnn tensor descriptor which used to define the meta data configuration info 
    cudnnTensorDescriptor_t tensor_desc; 
    cudnnCreateTensorDescriptor(&tensor_desc); 
    // here the CUDNN_TENSOR_NCHW tells the cudnn descriptor initializer following parameters order is:
    //  N(batch size) C(channel num) H(height) W(width)
    cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, 
                                CUDNN_DATA_FLOAT, batch_size, channels, height, width); 

    
    float alpha = 1.0f; 
    float beta  = 0.0f; 
    cudnnHandle_t cudnn = (*instance).getCudnnHandle(); 

    EXPECT_NE(&cudnn, nullptr); 

    // Invoke forward pass for activation: y = α × op(x) + β × y
    // activation_desc provides the 'op' function's implementation (e.g., sigmoid, ReLU, etc.)
    // tensor_desc provides the meta info like the number of rows and columns of the input and output matrices
    // input provides the 'x' matrix
    // alpha provides the value of 'α'
    // beta provides the value of 'β'
    // output provides the 'y' matrix
    cudnnActivationForward(
                cudnn, act_desc,
                &alpha, tensor_desc, input, 
                &beta, tensor_desc, output
                ); 

    // here print and verify the output of the forward pass 
    std::cout << "Forward pass output:" << std::endl; 
    for (int i = 0; i < size; i++) {
        std::cout << output[i] << " " << std::endl; 
    }                
    std::cout << std::endl; 
}


TEST_F(CudnnActivationTest, ApplySigmoidActivationBackforward)
{
    cudnnActivationMode_t mode      = CUDNN_ACTIVATION_SIGMOID; 
    cudnnNanPropagation_t nan_prop  = CUDNN_PROPAGATE_NAN; 
    double coef                     = 0.0; 
    cudnnActivationDescriptor_t act_desc = (*instance).getActivationDescriptor(); 


    const int batch_size = 2; 
    const int channels   = 3; 
    const int height     = 4; 
    const int width      = 4; 
    const int size       = batch_size * channels * height * width ; 

    float input[size] = {
                            1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 2.1, -2.1, 0.2, -0.2, 2.2, -2.2,
                            1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 2.1, -2.1, 0.2, -0.2, 2.2, -2.2,
                            1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.1, -0.1, 2.1, -2.1, 0.2, -0.2, 2.2, -2.2
                        };

    float expected[size] = {
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                        };  
    float output[size]; 
    float grad_input[size]; 
    float grad_output[size];                                               

    // create & cudnn tensor descriptor which used to define the meta data configuration info 
    cudnnTensorDescriptor_t tensor_desc; 
    cudnnCreateTensorDescriptor(&tensor_desc); 
    // here the CUDNN_TENSOR_NCHW tells the cudnn descriptor initializer following parameters order is:
    //  N(batch size) C(channel num) H(height) W(width)
    cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, 
                                CUDNN_DATA_FLOAT, batch_size, channels, height, width); 

    
    float alpha = 1.0f; 
    float beta  = 0.0f; 
    cudnnHandle_t cudnn = (*instance).getCudnnHandle(); 

    EXPECT_NE(&cudnn, nullptr); 

    // Invoke forward pass for activation: y = α × op(x) + β × y
    // activation_desc provides the 'op' function's implementation (e.g., sigmoid, ReLU, etc.)
    // tensor_desc provides the meta info like the number of rows and columns of the input and output matrices
    // input provides the 'x' matrix
    // alpha provides the value of 'α'
    // beta provides the value of 'β'
    // output provides the 'y' matrix
    cudnnActivationForward(
                cudnn, act_desc,
                &alpha, tensor_desc, input, 
                &beta, tensor_desc, output
                ); 

    // here print and verify the output of the forward pass 
    std::cout << "Forward pass output:" << std::endl; 
    for (int i = 0; i < size; i++) {
        std::cout << output[i] << " " ; 
    }                
    std::cout << std::endl; 


    // here we invoke loss function MES to calculate the loss value 
    float delta = 1.0; 
    double loss = (*instance).huberLoss(output, expected, delta, size); 
    std::cout << "Loss Value " << loss << std::endl; 

    // here we invoke gradient of loss function to calculate the gradient value array 
    (*instance).huberLossDerivative(output, expected, grad_output, delta, size);
    std::cout << "Huber Loss Gradient output:" << std::endl;  
    for (int i = 0; i < size; i++) {
        std::cout << grad_output[i] << " "; 
    }
    std::cout << std::endl; 

    cudnnDestroyTensorDescriptor(tensor_desc);

    // since we already have grad_output and input 
    // let's invoke the backforward operation 
    cudnnActivationBackward(
        cudnn, act_desc,
        &alpha, tensor_desc, output, tensor_desc, grad_output,
        tensor_desc, input, &beta, tensor_desc, grad_input
    );

    // here we print the grad_input value 
    std::cout << "Backward pass output (gradient w.r.t input): " << std::endl; 
    for (int i = 0; i < size; i++) {
        std::cout << grad_input[i] << " "; 
    }
    std::cout << std::endl; 
}
#include 

// helper class to wrap CudnnTensor basic operaitons 
class CudnnTensorHelper {
public: 
    CudnnTensorHelper(int n, int c, int h, int w):n_(n), c_(c), h_(h), w_(w) {
        // init cudnn 
        cudnnStatus_t status = cudnnCreate(&cudnn_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuDNN handle"); 
        }

        // create X and Y tensor descriptors' instances
        status = cudnnCreateTensorDescriptor(&x_tensor_desc_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroy(cudnn_);
            throw std::runtime_error("Failed to create cuDNN X tensor descriptor"); 
        }

        status = cudnnCreateTensorDescriptor(&y_tensor_desc_);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(x_tensor_desc_); 
            cudnnDestroy(cudnn_); 
            throw std::runtime_error("Failed to create cuDNN Y tensor descriptor"); 
        }


        // init X and Y tensor descriptors' instance s
        status = cudnnSetTensor4dDescriptor(x_tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            n_, c_, h_, w_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(x_tensor_desc_); 
            cudnnDestroy(cudnn_); 
            throw std::runtime_error("Failed to init cuDNN X tensor descriptor"); 
        }                                            

        status = cudnnSetTensor4dDescriptor(y_tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                            n_, c_, h_, w_); 
        if (status != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyTensorDescriptor(x_tensor_desc_); 
            cudnnDestroyTensorDescriptor(y_tensor_desc_); 
            cudnnDestroy(cudnn_); 
            throw std::runtime_error("Failed to init cuDNN X tensor descriptor"); 
        }                                             
    }

    ~CudnnTensorHelper() {
        cudnnDestroyTensorDescriptor(x_tensor_desc_); 
        cudnnDestroyTensorDescriptor(y_tensor_desc_); 
        cudnnDestroy(cudnn_); 
    }
private:
    cudnnHandle_t cudnn_; 
    cudnnTensorDescriptor_t x_tensor_desc_, y_tensor_desc_; 
    int n_; 
    int c_; 
    int h_; 
    int w_; 
}; 

// Google Test Driver class 
// helps add extra init and tear down operaiton 
class CudnnTensorTest : public ::testing::Test {
protected:
    static void SetUpTestCase() {
        // create 
        instance = new CudnnTensorHelper(); 
        
        // init 
        // todo 
    }

    static void TearDownTestCase() {
        // delete 
        delete instance; 

        // reset 
        instance = nullptr; 
    }

    static CudnnTensorHelper* instance; 
}; 

CudnnTensorHelper* CudnnTensorTest::instance = nullptr; 

TEST_F(CudnnTensorTest, CreateCudnnTensor) 
{

}
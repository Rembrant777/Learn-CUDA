#include <gtest/gtest.h>
#include <cudnn_activation.h>

using namespace std; 
using namespace testing; 

class CudnnActivationInstanceTest : public Test  {
public:
    CudnnActivationInstanceTest(): activaiton(nullptr) {}
    ~CudnnActivationInstanceTest() {}
    CudnnActivationInstance *activation;   

protected:
    virtual void SetUp() {
        cout << "before test" << endl; 
        activation = new CudnnActivationInstance(); 
    }    

    virtual void TearDown() {
        cout << "after test" << endl; 
        delete activation; // Free the memory allocated in SetUp
    } 
}; 


TEST_F(CudnnActivationInstanceTest, SetInvalidActivationDescriptor) 
{
    cudnnActivationMode_t mode = static_cast<cudnnActivationMode_t>(-1); 
    doule coef = 0.0; 

    EXPECT_NO_THROW(activation.set_activation_descriptor(mode, coef), std::runtime_error); 
}

int main(int argc, char **argv) 
{
     InitGoogleTest(&argc, argv); 
    return RUN_ALL_TESTS(); 
}


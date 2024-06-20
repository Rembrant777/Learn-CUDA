#include <gtest/gtest.h>
#include <hello_cudnn.h>

// ./test_hello_cudnn --gtest_filter=HelloCUDNNTest.*

// ./test_hello_cudnn --gtest_filter=HelloCUDNNTest.SimpleTest

TEST(HelloCUDNNTest, SimpleTest) 
{
    EXPECT_EQ(1, 1); 
}

// ./test_hello_cudnn --gtest_filter=HelloCUDNNTest.SayHello
TEST(HelloCUDNNTest, SayHello)
{
    HelloCudnn hello; 
    
    EXPECT_EQ("Hello cuDNN", hello); 
}



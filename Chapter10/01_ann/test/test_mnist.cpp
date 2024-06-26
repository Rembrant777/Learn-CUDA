#include "../src/mnist.h"
#include <gtest/gtest.h>

using namespace std; 
using namespace cudl; 

TEST(TestMnist, TestCreateAndInitMnist) {
    MNIST* mnist = new MNIST("../dataset"); 
    EXPECT_NE(mnist, nullptr); 
    delete mnist; 
}
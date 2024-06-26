#include "../src/mnist.h"
#include <gtest/gtest.h>

using namespace std; 
using namespace cudl; 

TEST(TestMnist, TestCreateAndInitMnist) {
    Mnist* mnist = new Mnist("../dataset"); 
    EXPECT_NE(mnist, nullptr); 
}
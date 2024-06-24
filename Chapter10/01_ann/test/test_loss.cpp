#include "../src/loss.h"
#include <gtest/gtest.h>

using namespace cudl; 

TEST(TestLoss, TestLossCreateAndDestroy) {
    CrossEntropyLoss* c_loss = new CrossEntropyLoss(); 
    EXPECT_NE(c_loss, nullptr); 
    EXPECT_NE(c_loss->get_d_loss(), nullptr); 
    EXPECT_NE(c_loss->get_h_loss(), 1.f);
    EXPECT_EQ(c_loss->get_d_workspace(), nullptr);  

    int batch_size = 8; 
    c_loss->init_workspace(batch_size); 
    EXPECT_NE(c_loss->get_d_workspace(), nullptr); 

    delete c_loss; 
}
TEST(TestLoss, clipFuncTest) {
    EXPECT_EQ(1, 1); 
}

TEST(TestLoss, softmaxLossKernelFuncTest) {
    EXPECT_EQ(1, 1); 
}

TEST(TestLoss, lossFuncTest) {
    EXPECT_EQ(1, 1); 
}
#include <gtest/gtest.h>

TEST(HelloGTest, EqualityTest) {
    EXPECT_EQ(1, 1);
    EXPECT_EQ(2 + 2, 4);
}

TEST(HelloGTest, InequalityTest) {
    EXPECT_NE(1, 2);
    EXPECT_NE(2 * 2, 5);
}


TEST(HelloGTest, BooleanTest) {
    EXPECT_TRUE(true);
    EXPECT_FALSE(false);
}

TEST(HelloGTest, FloatingPointTest) {
    EXPECT_FLOAT_EQ(3.0, 3.0);
    EXPECT_DOUBLE_EQ(0.1 + 0.2, 0.3);
    EXPECT_NEAR(0.1 + 0.2, 0.3, 1e-5);
}

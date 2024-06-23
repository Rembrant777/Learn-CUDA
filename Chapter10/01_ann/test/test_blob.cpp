#include "../src/blob.h"
#include <gtest/gtest.h>
using namespace cudl; 

TEST(TestBlob, TestBlobWithFloatType) {
    int n = 1, c = 2, h = 3, w = 4; 
    Blob<float>* b = new Blob<float>(n, c, h, w);
    EXPECT_NE(b, nullptr); 
    int EXPECT_LEN = n * c * h * w; 
    int EXPECT_SIZE = c * h * w; 
    int EXPECT_BUF_SIZE = EXPECT_LEN * sizeof(float); 

    EXPECT_EQ(b->size(), EXPECT_LEN);
    EXPECT_EQ(b->size(), EXPECT_SIZE);
    EXPECT_EQ(b->buf_size(), EXPECT_BUF_SIZE); 

    EXPECT_EQ(b->n(), n); 
    EXPECT_EQ(b->c(), c); 
    EXPECT_EQ(b->w(), w); 
    EXPECT_EQ(b->h(), h); 

    std::array<int, 4> reset_arr = {4,5,6,7};
    b->reset(reset_arr);  
    EXPECT_SIZE = 5 * 6 * 7; 
    EXPECT_LEN  = 4 * 5 * 6 * 7; 
    EXPECT_BUF_SIZE = EXPECT_LEN * sizeof(float);
    EXPECT_EQ(b->len(), EXPECT_LEN); 
    EXPECT_EQ(b->size(), EXPECT_SIZE); 
    EXPECT_EQ(b->buf_size(), EXPECT_BUF_SIZE); 
    EXPECT_EQ(b->n(), 4); 
    EXPECT_EQ(b->c(), 5); 
    EXPECT_EQ(b->h(), 6); 
    EXPECT_EQ(b->w(), 7); 

    // invoke tensor 
    EXPECT_EQ(false, b->is_tensor_); 
    b->tensor(); 
    EXPECT_EQ(true, b->is_tensor_); 

    float* host_ptr = b->ptr(); 
    EXPECT_NE(nullptr, host_ptr); 

    float* cuda_ptr = b->cuda(); 
    EXPECT_NE(nullptr, cuda_ptr); 

    delete b; 
} 

TEST(TestBlob, TestBlobInitWithIntType) {
    int n = 1, c = 2, h = 3, w = 4; 
    Blob<int>* b = new Blob<int>(n, c, h, w); 
    EXPECT_NE(b, nullptr); 
    int EXPECT_LEN = n * c * h * w; 
    int EXPECT_SIZE = c * h * w; 
    int EXPECT_BUF_SIZE = EXPECT_LEN * sizeof(int); 

    EXPECT_EQ(b->len(), EXPECT_LEN);
    EXPECT_EQ(b->size(), EXPECT_SIZE);
    EXPECT_EQ(b->buf_size(), EXPECT_BUF_SIZE); 

    EXPECT_EQ(b->n(), n); 
    EXPECT_EQ(b->c(), c); 
    EXPECT_EQ(b->w(), w); 
    EXPECT_EQ(b->h(), h); 

    std::array<int, 4> reset_arr = {4,5,6,7};
    b->reset(reset_arr);  
    EXPECT_LEN = 4 * 5 * 6 * 7; 
    EXPECT_SIZE  = 5 * 6 * 7; 
    EXPECT_BUF_SIZE = EXPECT_LEN * sizeof(int);
    EXPECT_EQ(b->len(), EXPECT_LEN); 
    EXPECT_EQ(b->size(), EXPECT_SIZE); 
    EXPECT_EQ(b->buf_size(), EXPECT_BUF_SIZE); 
    EXPECT_EQ(b->n(), 4); 
    EXPECT_EQ(b->c(), 5); 
    EXPECT_EQ(b->h(), 6); 
    EXPECT_EQ(b->w(), 7); 

    // invoke tensor 
    EXPECT_EQ(false, b->is_tensor_); 
    b->tensor(); 
    EXPECT_EQ(true, b->is_tensor_); 

    int* host_ptr = b->ptr(); 
    EXPECT_NE(nullptr, host_ptr); 

    int* cuda_ptr = b->cuda(); 
    EXPECT_NE(nullptr, cuda_ptr); 

    delete b;
}

TEST(TestBlob, TestBlobInitWithDoubleType) {
    int n = 1, c = 2, h = 3, w = 4; 
    Blob<double>* b = new Blob<double>(n, c, h, w); 
    EXPECT_NE(b, nullptr); 
    int EXPECT_LEN = n * c * h * w; 
    int EXPECT_SIZE = c * h * w; 
    int EXPECT_BUF_SIZE = EXPECT_LEN * sizeof(double); 

    EXPECT_EQ(b->len(), EXPECT_LEN);
    EXPECT_EQ(b->size(), EXPECT_SIZE);
    EXPECT_EQ(b->buf_size(), EXPECT_BUF_SIZE); 

    EXPECT_EQ(b->n(), n); 
    EXPECT_EQ(b->c(), c); 
    EXPECT_EQ(b->w(), w); 
    EXPECT_EQ(b->h(), h); 

    std::array<int, 4> reset_arr = {4,5,6,7};
    b->reset(reset_arr);  
    EXPECT_LEN = 4 * 5 * 6 * 7; 
    EXPECT_SIZE  = 5 * 6 * 7; 
    EXPECT_BUF_SIZE = EXPECT_LEN * sizeof(double);
    EXPECT_EQ(b->len(), EXPECT_LEN); 
    EXPECT_EQ(b->size(), EXPECT_SIZE); 
    EXPECT_EQ(b->buf_size(), EXPECT_BUF_SIZE); 
    EXPECT_EQ(b->n(), 4); 
    EXPECT_EQ(b->c(), 5); 
    EXPECT_EQ(b->h(), 6); 
    EXPECT_EQ(b->w(), 7); 

    // invoke tensor 
    EXPECT_EQ(false, b->is_tensor_); 
    b->tensor(); 
    EXPECT_EQ(true, b->is_tensor_); 

    double* host_ptr = b->ptr(); 
    EXPECT_NE(nullptr, host_ptr); 

    double* cuda_ptr = b->cuda(); 
    EXPECT_NE(nullptr, cuda_ptr); 

    delete b;
}

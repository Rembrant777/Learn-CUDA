#include <gtest/gtest.h>
// #include <gmock/gmock.h>

extern "C" {
    #include <cuda_runtime.h>
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

#include "../src/helper.h"

// class MockCudaContext {
// public:
//     MOCK_METHOD(cudaError_t, cudaMalloc, (void **, size_t)); 
//     MOCK_METHOD(const char*, cudaGetErrorString, (cudaError_t)); 
// }; 

void dummyCudaFunction(cudaError_t err) {
    checkCudaErrors(err); 
}

// test via redirect stderr to file 
TEST(HelperMacroFunction, checkCudaErrorsTestViaStderr) {
    int stderr_fd = dup(fileno(stderr)); 
    int temp_fd  = open("temp-stderr.log", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR); 
    dup2(temp_fd, fileno(stderr)); 

    // test with cudaErrorInvalidValue 
    dummyCudaFunction(cudaErrorInvalidValue); 

    // restore stderr 
    dup2(stderr_fd, fileno(stderr));
    close(temp_fd); 

    // read redirect file contents to buffer 
    FILE *file = fopen("temp-stderr.log", "r"); 
    char buffer[256]; 
    fgets(buffer, sizeof(buffer), file); 
    fclose(file); 

    // here checkout the output content 
    EXPECT_NE(strstr(buffer, "checkCudaErrors() API error"), nullptr); 

    // clean up stderr file 
    remove("temp-stderr.log"); 
}

// test via google mock apis 
// TEST(HelperMacroFunction, checkCudaErrorsTestViaMock) {
//     MockCudaContext mockContext; 

//     // here we define invoke mock cuda apis pre-defined cuda error enum will be thrown 
//     EXPECT_CALL(mockContext, cudaMalloc(::testing::_, ::testing::_))
//         .WillOnce(::testing::Return(cudaErrorInvalidValue));

//     EXPECT_CALL(mockContext, cudaGetErrorString(cudaErrorInvalidValue))        
//         .WillOnce(::testing::Return("cudaErrorInvalidValue")); 

//     // here begin invoke cuda apis 
//     void* ptr = nullptr; 
//     cudaError_t err = mockContext.cudaMalloc(&ptr, 100);         

//     // here we redirect stderr to a file 
//     int stderr_fd = dup(fileno(stderr)); 
//     int temp_fd   = open("temp_stderr.log", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR); 
//     dup2(temp_fd, fileno(stderr)); 

//     // here we call macro function defined in helper.h
//     checkCudaErrors(err); 

//     // here we restore the stderr 
//     dup2(stderr_fd, fileno(stderr)); 
//     close(temp_fd);

//     // here we read the contents of the file 
//     FILE* file = fopen("temp_stderr.log", "r"); 
//     char buffer[256]; 
//     fgets(buffer, sizeof(buffer), file); 
//     fclose(file); 

//     // here checkout the output content 
//     EXPECT_NE(strstr(buffer, "checkCudaErrors() API error"), nullptr); 

//     // clean up 
//     remove("temp_stderr.log"); 
// }
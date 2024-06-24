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

using namespace cudl; 

// class MockCudaContext {
// public:
//     MOCK_METHOD(cudaError_t, cudaMalloc, (void **, size_t)); 
//     MOCK_METHOD(const char*, cudaGetErrorString, (cudaError_t)); 
// }; 

void dummyCudaFunction(cudaError_t err) {
    checkCudaErrors(err); 
}

void dummyCublasFunction(cublasStatus_t err) {
    checkCublasErrors(err); 
}

void dummyCudnnFunction(cudnnStatus_t err) {
    checkCudnnErrors(err); 
}

// test via redirect stderr to file 

TEST(HelperMacroFunction, cublasGetErrorEnumTest) {
    std::vector<cublasStatus_t> cublasStatusList = {CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED,
                CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH,
                CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR,
                CUBLAS_STATUS_NOT_SUPPORTED, CUBLAS_STATUS_LICENSE_ERROR};
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[0]), "CUBLAS_STATUS_SUCCESS");                 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[1]), "CUBLAS_STATUS_NOT_INITIALIZED"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[2]), "CUBLAS_STATUS_ALLOC_FAILED"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[3]), "CUBLAS_STATUS_INVALID_VALUE"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[4]), "CUBLAS_STATUS_ARCH_MISMATCH"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[5]), "CUBLAS_STATUS_MAPPING_ERROR"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[6]), "CUBLAS_STATUS_EXECUTION_FAILED"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[7]), "CUBLAS_STATUS_INTERNAL_ERROR"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[8]), "CUBLAS_STATUS_NOT_SUPPORTED"); 
    EXPECT_EQ(_cublasGetErrorEnum(cublasStatusList[9]), "CUBLAS_STATUS_LICENSE_ERROR"); 
   
    EXPECT_NE(_cublasGetErrorEnum(cublasStatusList[9]), "<unknown>"); 
}

TEST(HelperMacroFunction, checkCudaErrorsTest) {
    int stderr_fd = dup(fileno(stderr)); 
    int temp_fd  = open("cuda-stderr.log", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR); 
    dup2(temp_fd, fileno(stderr)); 

    // test with cudaErrorInvalidValue 
    dummyCudaFunction(cudaErrorInvalidValue); 

    // restore stderr 
    dup2(stderr_fd, fileno(stderr));
    close(temp_fd); 

    // read redirect file contents to buffer 
    FILE *file = fopen("cuda-stderr.log", "r"); 
    char buffer[256]; 
    fgets(buffer, sizeof(buffer), file); 
    fclose(file); 

    // here checkout the output content 
    EXPECT_NE(strstr(buffer, "checkCudaErrors() API error"), nullptr); 

    // clean up stderr file 
    remove("cuda-stderr.log"); 

    stderr_fd = dup(fileno(stderr)); 
    temp_fd   = open("cublas-stderr.log", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR); 
    dup2(temp_fd, fileno(stderr)); 

    // test with cublas invalid status 
    dummyCublasFunction(CUBLAS_STATUS_INVALID_VALUE); 

    // restore stderr 
    dup2(stderr_fd, fileno(stderr));
    close(temp_fd);

    // read redirect file contents to buffer 
    file = fopen("cublas-stderr.log", "r"); 
    buffer[256]; 
    fgets(buffer, sizeof(buffer), file);
    fclose(file); 

    // here checkout the output content 
    EXPECT_NE(strstr(buffer, "checkCublasErrors() API error"), nullptr); 

    // clean stderr file 
    remove("cublas-stderr.log"); 

    stderr_fd = dup(fileno(stderr)); 
    temp_fd   = open("cudnn-stderr.log", O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR); 
    dup2(temp_fd, fileno(stderr)); 

    // test with cublas invalid status 
    dummyCudnnFunction(CUDNN_STATUS_NOT_INITIALIZED); 

    // restore stderr 
    dup2(stderr_fd, fileno(stderr));
    close(temp_fd);

    // read redirect file contents to buffer 
    file = fopen("cudnn-stderr.log", "r"); 
    buffer[256]; 
    fgets(buffer, sizeof(buffer), file);
    fclose(file); 

    // here checkout the output content 
    EXPECT_NE(strstr(buffer, "checkCudnnErrors() API error"), nullptr); 

    // clean stderr file 
    remove("cudnn-stderr.log"); 
}

// test via google mock apis (sad story TAT/~ gmock compiler not match with gpu's nvcc )
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

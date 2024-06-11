#include <cstdio>

using namespace std; 

__global__ 
void foo_kernel(int step) 
{
    printf("loop: %d\n", step); 
}

int main() 
{
    int n_stream = 5; 
    cudaStream_t *ls_stream; 
    ls_stream = (cudaStream_t*) new cudaStream_t[n_stream]; 

    // create multiple streams 
    for (int i = 0; i < n_stream; i++) {
        cudaStreamCreate(&ls_stream[i]); 
    }

    // execute kernels with the CUDA streame each 
    for (int i = 0; i < n_stream; i++) {
        foo_kernel<<<1,1,0, ls_stream[i]>>>(i); 
        cudaStreamSynchronize(ls_stream[i]); 
    }

    // synchornize the host and GPU 
    cudaDeviceSynchronize(); 

    // terminates all the created CUDA streams 
    for (int i = 0; i < n_stream; i++) {
        cudaStreamDestroy(ls_stream[i]); 
    }

    delete [] ls_stream; 

    return 0; 
}
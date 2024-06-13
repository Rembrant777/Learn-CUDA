# How Sgemm is Optimized 
![Diagram of `C = AB` ](https://private-user-images.githubusercontent.com/61581888/339216277-b5190968-244d-4215-9812-f22734ef166e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTgyNTQ2OTYsIm5iZiI6MTcxODI1NDM5NiwicGF0aCI6Ii82MTU4MTg4OC8zMzkyMTYyNzctYjUxOTA5NjgtMjQ0ZC00MjE1LTk4MTItZjIyNzM0ZWYxNjZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjEzVDA0NTMxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjMzlhMjgxYmRmNmUxNWE5YzA5ZmE0ZmEyYWUyMWQzNjAwODQ4ZDRkM2ZiNWIyNmY3ZTIyMzBjZmFhYWQ3MTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.01zt5DtxhA8opzdce5u7igBEcM9CTKxV5nKxngyibhE)

## Optimized Codes in `sgemm.cu`
We already know that 
* `A` is a matrix with dim (M * K).
* `B` is a matrix with dim (K * N).
* Result `C` is a matrix with dim (M * K)

I have to say that one quesiton that confused me for a long time is:
Even though CUDA defines the dimension in 3D which not means the CUDA's correpoinding blocks or GPU cores are organized in cube. It defines as 3D just want to define different dimensions of {x, y, z} as different unit of parallelism easy to distinguish(and also easy to get confused for beginner TAT/~ ). 


```cuda
#define BLOCK_DIM 16   

__global__
void sgemm_kernel_opt_version(const float *a, const float *B, float *C,
            int M, int N, int K, float alpha, float beta)
{
    int bid_x = blockIdx.x * blockDim.x; 
    int bid_y = blockIdx.y * blockDim.y; 
    int tid_x = threadIdx.x; 
    int tid_y = threadIdx.y; 

    float element_c = 0.f; 
    __shared__ float s_title_A[BLOCK_DIM][BLOCK_DIM]; 
    __shared__ float s_title_B[BLOCK_DIM][BLOCK_DIM]; 

    // forward tile with size in matrix A
    for (int k = 0; k < K)

}            
```
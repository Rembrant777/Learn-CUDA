# How Sgemm is Optimized 
![Diagram of `C = AB` ](https://private-user-images.githubusercontent.com/61581888/339216277-b5190968-244d-4215-9812-f22734ef166e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTgyNTQ2OTYsIm5iZiI6MTcxODI1NDM5NiwicGF0aCI6Ii82MTU4MTg4OC8zMzkyMTYyNzctYjUxOTA5NjgtMjQ0ZC00MjE1LTk4MTItZjIyNzM0ZWYxNjZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjEzVDA0NTMxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjMzlhMjgxYmRmNmUxNWE5YzA5ZmE0ZmEyYWUyMWQzNjAwODQ4ZDRkM2ZiNWIyNmY3ZTIyMzBjZmFhYWQ3MTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.01zt5DtxhA8opzdce5u7igBEcM9CTKxV5nKxngyibhE)

## Optimized Codes in `sgemm.cu`
We already know that 
* `A` is a matrix with dim (M * K).
* `B` is a matrix with dim (K * N).
* Result `C` is a matrix with dim (M * K)

I have to say that one quesiton that confused me for a long time is:
Even though CUDA defines the dimension in 3D which not means the CUDA's correpoinding blocks or GPU cores are organized in cube. It defines as 3D just want to define different dimensions of {x, y, z} as different unit of parallelism easy to distinguish(and also easy to get confused for beginner TAT/~ ). 

## Different Hierarchy Compute Components in CUDA
* Hirarchy of compute components defined in CUDA 
```
Grid > Block > Thread
```
> Grid: The highest level of organizaiton. A grid contians multiple blocks. 
> Block: A grid is divided into blocks. Each block contains multiple threads. 
> Thread: The smallest unit of computation within a block. 

* Fine-Grained Management Concepts in CUDA 
```
Grid > Block > (Warp) > Thread 
```



* To provide more fine-grined management import some logical concepts
Grid > Group > Block > Warp(tile) > Thread 

## Understand Different Types of Index in CUDA 
### `GridIdx`
### `BlockIdx`
* `blockIdx`: This is CUDA inner defined 3 dim variable, which includes current thread block(Block)'s index in the scope of Grid. 

* `blockIdx.x`: Current Block's index in the X axis of Grid. 
* `blockIdx.y`: Current Block's index in the X axis of Grid. 

### `BlockDim`
* `blockDim`: This is a CUDA inner defined 3 dim variable, which determines how many threads per block. 
* `blockDim.x`: How many threads available on the X axis in current Block.
* `blockDim.y`: How many threads available on the Y axis in current Block. 

### `ThreadIdx`
* `threadIdx`: CUDA 
* `threadIdx.x`
* `threadIdx.y`


* `BLOCK_DIM`: dimension of the thread block which means block thread dimension is 16, each block has 16 * 16 = 256 in total threads. 

* `blockIdx.x`: means one block's thread index in the dimension of X axis. Since we set the `BLOCK_DIM` as 16, so we know that in the scope of block `blockIdx.x` value should be in the range of `[0, 16 - 1]`.

* `blockIdx.y`: means one block's thread index in the dimension of Y axis. Since we set the `BLOCK_DIM` as 16, so we know that in the scope of block `blockIdx.y` value should be in the range of `[0, 16 - 1]`.



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
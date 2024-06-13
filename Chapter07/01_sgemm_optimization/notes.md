# How Sgemm is Optimized 
![Diagram of `C = AB` ](https://private-user-images.githubusercontent.com/61581888/339216277-b5190968-244d-4215-9812-f22734ef166e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTgyNTQ2OTYsIm5iZiI6MTcxODI1NDM5NiwicGF0aCI6Ii82MTU4MTg4OC8zMzkyMTYyNzctYjUxOTA5NjgtMjQ0ZC00MjE1LTk4MTItZjIyNzM0ZWYxNjZlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjEzVDA0NTMxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJjMzlhMjgxYmRmNmUxNWE5YzA5ZmE0ZmEyYWUyMWQzNjAwODQ4ZDRkM2ZiNWIyNmY3ZTIyMzBjZmFhYWQ3MTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.01zt5DtxhA8opzdce5u7igBEcM9CTKxV5nKxngyibhE)

## Optimized Codes in `sgemm.cu`
We already know that 
* `A` is a matrix with dim (M * K).
* `B` is a matrix with dim (K * N).
* Result `C` is a matrix with dim (M * K)

I have to say that one quesiton that confused me for a long time is:
Even though CUDA defines the dimension in 3D which not means the CUDA's correpoinding blocks or GPU cores are organized in cube. It defines as 3D just want to define different dimensions of {x, y, z} as different unit of parallelism easy to distinguish(and also easy to get confused for beginner TAT/~ ). 

### CUDA GPU Resource Logic Codes
Suppose `A[M][K] * B[K][N] = C[M][N]`.

We set M = 128 and N = 256, K = 64
```
A#height = M = 128
A#wide    = K = 64

B#height = K = 64
B#wide   = N = 256

C#height = M = 128
C#wide   = N = 256

BLOCK_DIM = 16 = Each Axis 16 Threads per Block

BLOCK Total Threads = Block.X Axis * Block.Y Axix = 256 Threads / Block


Then we got the sub-matrix of both _S_A and _S_B is (K * K) 

```cuda
// apply Block with 
// X axis ${BLOCK_DIM} = 16 threads 
// Y axis ${BLOCK_DIM} = 16 threads s
dim3 blockDim(BLOCK_DIM, BLOCK_DIM);

// apply Grid with 
// X axis ${(N + BLOCK_DIM -1) / BLOCK_DIM} = (256 + 16 - 1) / 16 = 16 Blocks
// Y axis ${(M + BLOCK -1)/ BLOCK_DIM} = (128 + 16 -1 ) / 16 = 8 Blocks 

dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, 
            (M + BLOCK_DIM - 1) / BLOCK_DIM);

// apply from Grid with Blocks X axis 16 , Y axis 8 Blocks 
// X axix 16 Blocks in charge of X dimension Matrix computation 
// Y axix 8 Blocks(with 256 Threads)
sgemm_kernel_opt_version<<<gridDim, blockDim>>>(...); 
```

#### CUDA Thread Execution Func 
```cuda
#define BLOCK_DIM 16   

__global__
void sgemm_kernel_opt_version(const float *a, const float *B, float *C,
            int M, int N, int K, float alpha, float beta)
{
    // bid_x represents the global init offset in globa Matrix A, B, and C 
    int bid_x = blockIdx.x * blockDim.x; 

    // bid_y represents the global init offset in global Matrix A, B, and C
    int bid_y = blockIdx.y * blockDim.y; 

    int tid_x = threadIdx.x; 
    int tid_y = threadIdx.y; 

    float element_c = 0.f;

    __shared__ float _S_A[BLOCK_DIM][BLOCK_DIM]; 
    __shared__ float _S_B[BLOCK_DIM][BLOCK_DIM]; 

    // forward tile with size in matrix A
    for (int k = 0; k < K; k += BLOCK_DIM) {

        // each thread copy data from global Matrix A, B 
        // to Block scoped Matrix _S_A, _S_B
        _S_A[tid_y][tid_x] = A[ (bid_y + tid_y) * K + tid_x + k ];
        _S_B[tid_y][tid_x] = B[ (k + tid_y) * N + bid_x + tid_x ];

        // here wait all threads in different blocks 
        // write global A,B matrixes values to the block-grained 
        // shared memory _S_A and _S_B 
        __syncthreads(); 

        // the for loop will also be executed by multiple threads from multiple blocks
        // each thread computes: sub-matrix values from _S_A, _S_B
        // _S_A[tid_y][0...BLOCK_DIM-1] * _S_B[0...BLOCK_DIM-1][tid_x]
        // compute gemm operation with tiles 
        for (int e = 0; e < BLOCK_DIM; e++) {
            element_c += _S_A[tid_y][e] * _S_B[e][tid_x];
        }

        // here wait all blocks' correspoinding threads execute the col * row operation
        // all mid results are write to the shared memroy 
        __syncthreads(); 
    }

    // here each thread in correspoinding blocks 
    // extract its value from the middle result matrix -> 
    // multiple middle result to parameters of {alpha, beta}
    // and write data to the global matrix C 
    // and the index of the global matrix C is calculated by 
    // current threadIdx.x, threadIdx.y and the BlockID and the Block Num 
    C[(bid_y + tid_y) * N + (bid_x + tid_x)] = \
            alpha * element_c + beta * C[(bid_y + tid_y) * N + (bid_x + tid_x)];
}            
```

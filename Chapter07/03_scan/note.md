# An Introduction of Prefix Sum Scan 

## Introduction 
Prefix Sum Scan is a kind of parallel algoritm that designed to calculate each elements in the array's prefix-sum. 
Suppose there is an array A, its prefix-sum array is called B.
Each element in `B[i]` represents the accumulation of elements' in the range of`A[0] + ... + A[i]`

```latex
The prefix sum for an array \( A \) is given by:

\[ B[i] = \sum_{j=0}^{i} A[j] \]
```
## Algorithm Execute Steps 
Algorihm implementation can be divided into two phase. 
* Reduction or Upsweep Phase
In this step, we accumulate elements in the array via stride to calculate its phase sum value.

* Downsweep Phase
In this step, we re-distribute previous phase calculated values to the array, and continue to calculate to get its prefix-sum values. 

## Algorithm Explanation 
### Phase-1 Reduction Phase
#### Step-1: set init value of `offset = 1`
#### Step-2: iterate 
* each thread manipulates two index `idx_a` and `idx_b` to index two elements in array `elem_a`, and `elem_b`
* accumulate value `elem_a` to `elem_b` and accumulated result overwrite to the location of `elem_b` which is `array[idx_b]`
* duplicate the value of `offset`, e.g., `offset <<= 1`

For example 
* init value `array[a0, a1, a2, a3, a4, a5, a6, a7]` and `offset = 1`
* step-1(offset=1) `[a0, a0 + a1, a2, a2 + a3, a4, a4 + a5, a6, a6 + a7]`
* step-2(offset=2) `[a0, a0+a1, a0+a1+a2+a3, a0+a1+a2+a3, a4, a4+a5, a4+a5+a6+a7, a4+a5+a6+a7]`
* step-3(offset=4) `[a0, a0+a1, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3+a4+a5+a6+a7, a0+a1+a2+a3+a4+a5+a6+a7, a0+a1+a2+a3+a4+a5+a6+a7, a0+a1+a2+a3+a4+a5+a6+a7]`

### Phase-2 Downsweep Phase
#### Step-1: set init value of `offset = length / 2`
#### Step-2: iterate
* each thread manipulates two index `idx_a` and `idx_b`
* accumulate element from `array[idx_b]` to `arr[idx_a]`
* divide the value of `offset`, e.g., `offset >>=1`

For example
* init value `array[a0, a0+a1, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3+a4+a5+a6+a7, a0+a1+a2+a3+a4+a5+a6+a7, a0+a1+a2+a3+a4+a5+a6+a7, a0+a1+a2+a3+a4+a5+a6+a7]` and `offset = 8 / 2 = 4`
* step-2(offset=4) `[a0, a0+a1, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3+a4+a5, a0+a1+a2+a3+a4+a5, a0+a1+a2+a3+a4+a5+a6+a7]`
* step-3(offset=2): `[a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3+a4, a0+a1+a2+a3+a4+a5, a0+a1+a2+a3+a4+a5+a6+a7]`
* step-4(offset=1): `[a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, a0+a1+a2+a3+a4, a0+a1+a2+a3+a4+a5, a0+a1+a2+a3+a4+a5+a6, a0+a1+a2+a3+a4+a5+a6+a7]`


## CUDA Implementation 
If we want to design this parallel algorithm upon CUDA pattern.  What we need to care about are two points:
1. scale of the array, decide the `Block` number and each `Thread` number per block. more than that, the scale of the array and the number of the `Block` also determins the grain of parallelism and the shared memory space. 

2. Points of thread sync in the algorithm. We need to design the main points to exeucte `__syncthreads` to sync the middle results to avoid thread conflict or data missing. 

### Example of CUDA Resource Allocation 

Suppose we need to execute hte prefix-sum scan upon an array with length of 128.
And we apply 2 blocks and each block apply 4 threads to process the 2 phase {reduction, downsweep}
Then for `block-0 `it will handle the array[0...63] and `block-1 ` it will handle the array[64 ...128]. 

For `block-1`, it has 4 threads with idx(threadIdx.x) from `[0 ... 3]`
All threads in `block-1` will share the shared memory that cache the data copied from global memory (`array[64 ... 128-1]`)
So the shared memory `_s_b[0 ... 64]`
For `block-1`'s `thread-0` it will handle data, we know that index of `idx_a` and `idx_b`
#### thread-0:
* iteration-0:(tid = 0; offset = 1) 
// array[1] = array[0] + array[1]
`idx_a = offset * (2 * tid + 1) - 1 = 1 * (2 * 0 + 1) - 1 = 0`
`idx_b = offset * (2 * tid + 2) - 1 = 1 * (2 * 0 + 2) - 1 = 1`
`offset <<= 1 -> 2`

* iteration-1:(tid = 0; offset = 2)
// array[2] = array[1] + array[2] 
`idx_a = offset * (2 * tid + 1) - 1 = 2 * (2 * 0 + 1) - 1 = 1`
`idx_b = offset * (2 * tid + 2) - 1 = 2 * (2 * 0 + 2) - 1 = 2` 
`offset <<= 1 -> 2`
* iteration-2:(tid = 0; offset = 4)
`...`

#### thread-1: 
* iteration-0: (tid = 1; offset = 1)
// array[4] += array[3] 
`idx_a = offset * (2 * tid + 1) - 1 = 1 * (2 * 1 + 1) - 1 = 2`
`idx_b = offset * (2 * tid + 2) - 1 = 1 * (2 * 1 + 2) - 1 = 3`
`offset <<= 1 -> 2`

#### thread-2: 
* iteration-0: (tid = 2; offset = 1)
`idx_a = offset * (2 * tid + 1) - 1 = 1 * (2 * 2 + 1) - 1 = 3`
`idx_b = offset * (2 * tid + 2) - 1 = 1 * (2 * 2 + 2) - 1 = 4`

#### thread-3: 
* iteration-0: (tid = 3; offset = 1)
`idx_a = offset * (2 * tid + 1) - 1 = 1 * (2 * 2 + 1) -1 = 4`
`idx_b = offset * (2 * tid + 2) - 1 = 1 * (2 * 2 + 2) -1 = 5`

## Reference 
* Book Learn CUDA Programming 
* ChatGPT TVT/~ 
# Chapter 05 CUDA Application Profiling and Debugging 


## How to profiling cuda binary file via cuda's `nvprof`
### Step-0 Add cuda profile api header 
Import cuda profiling header file to cuda source code.
```cuda
#include <cuda_profiler_api.h>
```

### Step-1 add cuda profile functions 
Add cuda profiling functions around the to be profiling codes. Like:
```cuda
cudaProfilerStart();
    for (int i = 0; i < n_iter; i++)
        sgemm_gpu_B(d_A, d_B, d_C, N, M, K, alpha, beta);
cudaProfilerStop();
```
### Step-2 invoke nvprof shell command 

* Common usage of `nvprof`
```shell
#!/bin/sh
export NVCC_HOME=/usr/local/cuda-9.2

${NVCC_HOME}/bin/nvprof -f -o profile-original.nvvp ./sgemm
```

* `nvprof` command that supports extra focused area options
Also can provide to the extra options like `nvprof` like `--profile-from-start off` to let the nvprof
focus on profiling the profiling functions embraced areas. 
This option helps you focus on the module you're currently developing, and lets you remove irrelevant operations
from the report in the profiler. 
```shell
#!/bin/sh
export NVCC_HOME=/usr/local/cuda-9.2

${NVCC_HOME}/bin/nvprof -f -o profile-original.nvvp --profile-from-start off ./sgemm
```

* `nvprof` command that supports limit time options 
The NVIDIA profile has other options that can limit profile targets. 

1. option-1: `--timeout <second>`
This option limits application execution time, and it is very useful especially when you need to profile an application that has a long execution time with iterative operations. 


2. option-2: `--devices <gpu ids>`
This option helps you narrow down GPU kernel operations in a multiple GPU application. 


3. other options like `--kernels` `--event` `--metrics`
If you do not have to collect all the metrics you just want to focuse on a few kernel functions. 
Use these options 

```shell
export NVCC_HOME=/usr/local/cuda-9.2

${NVCC_HOME}/bin/nvprof -f -o profile-original.nvvp \
    # this let the profiling tool(nvprof) only focus on the kernel function sgemm_kernel_B
    --kernels sgemm_kernel_B \  
    --metrics all ./sgemm
``` 

## How to profiling with NVTX 
### What's NVTX 
NVTX is NVIDIA Tools Extension which can provides developers a way to analyze functional performance in a complex application. 

### How to use NVTX
If we want to use NVTX to profiling CUDA functions, we can annotate the CUDA code like this:
```cuda
#include "nvToolsExt.h"
nvtxRangePushA("Annotation"); 
...
{
    Range of GPU Kernel operations
}
...
cudaDeviceSynchronization(); 

// here we define a range as a group of codes and annotate its range by manual.

// then here the CUDA profiler provides a timeline trace of the 
// annotation so that we can measure the execution time of code blocks. 
callsnvtxRangePop(); 
```

* Tips
```
One drawback of NVTX function is that its APIs are host(CPU side) functions.
So every time we need to sync the host and GPU data before we invoke this function. 
```

## NVTX Compile Extra Options
To compile the code, we should provide the `-lnvToolsExt` option to the nvcc compiler to provide NVTX API's definition. 
```shell
nvcc -m64 -gencode arch=compute_$70,code=sm_70 -lnvToolsExt -o sgemm sgemm.cu
```

## Use both NVTX and NVIDIA Profiler as Complementary  
NVIDIA profiler anc collect NVTX annotations without extra options. 
With the help of NVIDIA profiler, developer can shrink the focused code areas in more specific blocks.
And with the help of NVTX, developer can retrieve more fine-grain profile metrics. 

We can also profile the application using the following command.
```shell
nvprof -f --profile-from-start off -o sgemm.nvvp ./sgemm.nvvp 
```

For more information of NVTX refer to this [doc link](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx).
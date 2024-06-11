#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("This device does not support dynamic parallelism\n");
        return -1;
    }
    printf("This device supports dynamic parallelism\n");
    return 0;
}

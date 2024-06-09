# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=reduction_sequential reduction_interleaving

INCLUDES= -I${CUDA_PATH}/samples/common/inc -I${CUDA_PATH}/include
# here we use --resource-usage to enable GPU resource usage info 
NVCC_FLAGS=-m64 -lineinfo --resource-usage

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -lgomp
ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true $(NVCC_FLAGS)

all: ${TARGET}

reduction_kernel_interleaving.o: reduction_kernel_interleaving.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_kernel_sequential.o: reduction_kernel_sequential.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction.o: reduction.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_sequential: reduction.o reduction_kernel_sequential.o
	$(EXEC) $(NVCC) ${INCLUDES} -o $@ $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) $+

reduction_interleaving: reduction.o reduction_kernel_interleaving.o
	$(EXEC) $(NVCC) ${INCLUDES} -o $@ $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(LIBRARIES) $+

clean:
	rm -rf ${TARGET} *.o	

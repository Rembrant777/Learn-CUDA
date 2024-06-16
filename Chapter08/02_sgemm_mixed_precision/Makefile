# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cublasGemmEx

INCLUDES= -I${CUDA_PATH}/samples/common/inc 
NVCC_FLAGS=-m64 -lineinfo #--resource-usage

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -lcublas
ALL_CCFLAGS += -std=c++11 $(INCLUDES) -L$(CUDA_PATH)/lib64

all : ${TARGET}

cublasGemmEx: cublasGemmEx.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: cublasGemmEx
	$(CUDA_PATH)/bin/nvprof -f -o $+.nvvp ./$+	

clean:
	rm -f ${TARGET} *.o
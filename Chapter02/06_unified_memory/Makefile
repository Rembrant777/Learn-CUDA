# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=unified_simple unified_initialized unified_prefetch unified_64align

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all: ${TARGET}

unified_simple: unified_memory.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODEFLAGS} -o $@.out $<

unified_initialized: unified_memory_initialized.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODEFLAGS} -o $@.out $<

unified_prefetch: unified_memory_prefetch.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODEFLAGS} -o $@.out $<

unified_64align: unified_memory_64align.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@.out $<	

clean:
	rm -rf ${TARGET} && rm -rf *.out
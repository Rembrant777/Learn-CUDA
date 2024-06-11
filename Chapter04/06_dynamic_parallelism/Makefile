# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=recursion
#is_support_dynamic_paralleism dynamic_parallelism recursion
#dynamic_parallelism 

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
# SMS = 52 60 61 70 75 80
# ifeq "$(IS_CUDA_9)" "1"
# SMS = 35 52 60 61 70 75 80 86
# endif
SMS =35 
GENCODE_FLAGS = $(foreach sm, ${SMS}, -gencode arch=compute_$(sm),code=sm_$(sm))

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all: ${TARGET}

dynamic_parallelism: dynamic_parallelism.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODEFLAGS} -o $@.out $<

is_support_dynamic_paralleism: is_support_dynamic_paralleism.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODEFLAGS} -o $@.out $<

recursion: recursion.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODEFLAGS} -o $@.out $<	

clean:
	rm -rf ${TARGET} && rm -rf *.out
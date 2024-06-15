# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=pack_n_split

INCLUDES= -I${CUDA_PATH}/samples/common/inc 
# --lineinfo this opt enables nvcc compiler generates line number provides developers debug details 
# -m64 enables nvcc compiler utilize current os's memory and register resources 
NVCC_FLAGS=-m64 --resource-usage -lineinfo

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# here append cuda's library to compile commands to link 
# we can use the cuda library as long as we add the header files 
LIBRARIES += -L/usr/local/cuda/lib
ALL_CCFLAGS += -std=c++11  $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES)

all : ${TARGET}

scan_v2.o: scan_v2.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

pack_n_split: pack_n_split.cu scan_v2.o
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

nvprof: pack_n_split
	${CUDA_PATH}/bin/nvprof -f -o $+.nvvp --profile-from-start off ./$+
	${CUDA_PATH}/bin/nvprof -f -o $+-metrics.nvvp --analysis-metrics ./$+

clean:
	rm -f ${TARGET} *.o *.nvvp
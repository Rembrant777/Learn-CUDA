# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
MPICC ?= mpicc
TARGET=simpleMPI
INCLUDES= -I${CUDA_PATH}/samples/common/inc -I${CUDA_PATH}/include
# here we use --resource-usage to enable GPU resource usage info 
NVCC_FLAGS=-m64 -Xcompiler -fopenmp -rdc=true -lcudadevrt -lmpi  
IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp
ALL_CCFLAGS += -g -Xcompiler -fopenmp $(NVCC_FLAGS)

all : ${TARGET}

simpleMPI: simpleMPI.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

enable_mps:
	export CUDA_VERSION_DEVICES=0
	sudo nvidia-smi -c 3 -i 0
	sudo nvidia-cuda-mps-control -d 

nvprof: simpleMPI
	mpirun -np ${PROCS} nvprof -f -o $+.%q{OMPI_COMM_WORLD_RANK}_${STREAMS}.nvvp ./$+ ${STREAMS}

disable_mps:
	echo "quit" | sudo nvidia-cuda-mps-control
	sudo nvidia-smi -c 0 -i 0	


clean:
	rm -f ${TARGET} *.o *.nvvp
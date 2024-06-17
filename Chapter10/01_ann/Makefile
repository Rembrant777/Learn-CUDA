# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=train

INCLUDES= -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include
NVCC_FLAGS=-m64 --resource-usage -lineinfo

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L/usr/local/cuda/lib -lcublas -lcudnn -lgomp -lcurand
ALL_CCFLAGS += -m64 -g -std=c++11  $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES)

SRC_DIR = src 
OBJ_DIR = obj 

all : ${TARGET}

INCS = ${SRC_DIR}/helper.h ${SRC_DIR}/blob.h ${SRC_DIR}/blob.h ${SRC_DIR}/layer.h 

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp ${INCS}
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: ${SRC_DIR}/%.cu ${INCS}
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

$(OBJ_DIR)/train.o train.cpp ${INCS}
	@mkdir -p $(@D)
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

OBJS = ${OBJ_DIR}/train.o ${OBJ_DIR}/mnist.o ${OBJ_DIR}/loss.o ${OBJ_DIR}/layer.o ${OBJ_DIR}/network.o 

train: $(OBJS)
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

.PHONY: clean 	
clean:
	rm -f ${TARGET} ${OBJ_DIR}/*.o
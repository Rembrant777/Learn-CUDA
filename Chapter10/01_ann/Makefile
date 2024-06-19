# Here we declare the path of CUDA
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=train

INCLUDES= -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include
NVCC_FLAGS=-m64 --resource-usage -lineinfo

IS_CUDA_9:=$(shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2)

# Gencode arguments
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L$(CUDA_PATH)/lib64 -lcublas -lcudnn -lgomp -lcurand
ALL_CCFLAGS += -g -std=c++11 $(NVCC_FLAGS) $(INCLUDES)

SRC_DIR = src
OBJ_DIR = obj

all: ${TARGET}

INCS = ${SRC_DIR}/helper.h ${SRC_DIR}/blob.h ${SRC_DIR}/layer.h ${SRC_DIR}/loss.h ${SRC_DIR}/mnist.h ${SRC_DIR}/network.h

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/train.o: train.cpp ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

OBJS = ${OBJ_DIR}/train.o ${OBJ_DIR}/mnist.o ${OBJ_DIR}/loss.o ${OBJ_DIR}/layer.o ${OBJ_DIR}/network.o

train: $(OBJS)
	$(NVCC) $(ALL_CCFLAGS) $(OBJS) -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

.PHONY: clean
clean:
	rm -f ${TARGET} ${OBJ_DIR}/*.o

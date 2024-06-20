# Path of CUDA
CUDA_PATH=/usr/local/cuda-9.2
GOOGLETEST_PATH=/home/raytrack/workspace/Learn-CUDA/Chapter10/googletest
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TEST_TARGET=test_cudnn_04 

# Include paths
INCLUDES= -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include -I$(GOOGLETEST_PATH)/googletest/include -I$(GOOGLETEST_PATH)/googletest/

# Compiler flags
NVCC_FLAGS=-m64 --resource-usage -lineinfo -lnvToolsExt

# Determine if CUDA version is 9.2 or later
IS_CUDA_9:=$(shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2)

# Gencode arguments
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Libraries
LIBRARIES += -L$(CUDA_PATH)/lib64 -lcublas -lcudnn -lgomp -lcurand 

# Compiler flags
ALL_CCFLAGS += -g -std=c++14 $(NVCC_FLAGS) $(INCLUDES)

# Directories
OBJ_DIR = obj
TEST_DIR = test
GTEST_DIR = $(GOOGLETEST_PATH)

# All targets
all:  ${TEST_TARGET}

OBJ_FILES = $(OBJ_DIR)/test_cudnn_activation_descriptor.o $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o

${OBJ_DIR}/gtest-all.o: ${GTEST_DIR}/googletest/src/gtest-all.cc
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/gtest_main.o: ${GTEST_DIR}/googletest/src/gtest_main.cc
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_cudnn_activation_descriptor.o: ${TEST_DIR}/test_cudnn_activation_descriptor.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

test_cudnn_04: $(OBJ_FILES)
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)
	./$@

.PHONY: clean
clean:
	rm -f ${TEST_TARGET} ${OBJ_DIR}/*.o 

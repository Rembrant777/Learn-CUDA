# Path of CUDA
CUDA_PATH=/usr/local/cuda-9.2
GOOGLETEST_PATH=/home/raytrack/workspace/Learn-CUDA/Chapter10/googletest
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
#TARGET=hello_cudnn
TEST_TARGET=test_hello_cudnn

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
#-lgtest -lgtest_main  


# Compiler flags
ALL_CCFLAGS += -g -std=c++14 $(NVCC_FLAGS) $(INCLUDES)

# Directories
SRC_DIR = src
OBJ_DIR = obj
TEST_DIR = test
GTEST_DIR = $(GOOGLETEST_PATH)

# Headers
INCS = ${SRC_DIR}/hello_cudnn.h

# Object files
OBJS = ${OBJ_DIR}/hello_cudnn.o
GTEST_OBJS = ${OBJ_DIR}/gtest-all.o ${OBJ_DIR}/gtest_main.o

# All targets
all:  ${TARGET} ${TEST_TARGET}

# Rule for training target
${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/hello_cudnn.o: ${SRC_DIR}/hello_cudnn.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

# hello_cudnn: $(OBJS)
# 	$(NVCC) $(ALL_CCFLAGS) $(OBJS) -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

# Google Test rules
${OBJ_DIR}/gtest-all.o: ${GTEST_DIR}/googletest/src/gtest-all.cc
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/gtest_main.o: ${GTEST_DIR}/googletest/src/gtest_main.cc
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_hello_cudnn.o: ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)


# Test target
TEST_SRCS = ${TEST_DIR}/test_hello_cudnn.cpp 
TEST_OBJS = ${OBJ_DIR}/test_hello_cudnn.o 

test_hello_cudnn: ${TEST_OBJS} ${GTEST_OBJS}
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)
	./$@

# Clean rule
.PHONY: clean
clean:
	rm -f ${TARGET} ${TEST_TARGET} ${OBJ_DIR}/*.o 

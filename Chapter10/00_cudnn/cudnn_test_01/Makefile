# Path of CUDA
CUDA_PATH=/usr/local/cuda-9.2
GOOGLETEST_PATH=/home/raytrack/workspace/Learn-CUDA/Chapter10/googletest
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TEST_TARGET=test_cudnn_activation test_cudnn_activation_descriptor

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
INCS = ${SRC_DIR}/cudnn_activation.h

# Object files
OBJS = ${OBJ_DIR}/cudnn_activation.o
GTEST_OBJS = ${OBJ_DIR}/gtest-all.o ${OBJ_DIR}/gtest_main.o

# All targets
all:  ${TEST_TARGET}

# Rule for training target
${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/cudnn_activation.o: ${SRC_DIR}/cudnn_activation.cpp ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/cudnn_activation_descriptor.o: ${SRC_DIR}/cudnn_activation_descriptor.cpp ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)


GTEST_ALL_OBJ=${OBJ_DIR}/gtest-all.obj
GTEST_MAIN_OBJ=${OBJ_DIR}/gtest_main.o 

# Google Test rules, if obj files already exist just skip 
${OBJ_DIR}/gtest-all.o: ${GTEST_DIR}/googletest/src/gtest-all.cc
ifeq (, $(wildcard $(GTEST_ALL_OBJ)))	
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)
endif 	

${OBJ_DIR}/gtest_main.o: ${GTEST_DIR}/googletest/src/gtest_main.cc
ifeq (, $(wildcard $(GTEST_MAIN_OBJ)))
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)
endif	

${OBJ_DIR}/test_cudnn_activation.o: ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_cudnn_activation_descriptor.o: ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

test_cudnn_activation: ${OBJ_DIR}/test_cudnn_activation.o ${GTEST_OBJS}
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)
	./$@

test_cudnn_activation_descriptor: ${OBJ_DIR}/test_cudnn_activation_descriptor.o ${GTEST_OBJS}
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)
	./$@

# Clean rule
.PHONY: clean

# finds all google objects and skip remove them during make clean period 
# to avoid re-compile gtest objects during make period
clean:
	find ${OBJ_DIR} -type f -name '*.o' ! -name 'gtest-all.o' ! -name 'gtest_main.o' -exec rm -f {} +
	rm -f ${TEST_TARGET}

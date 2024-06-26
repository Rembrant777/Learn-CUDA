# Here we declare the path of CUDA
CUDA_PATH=/usr/local/cuda-9.2
GOOGLETEST_PATH=/home/raytrack/workspace/Learn-CUDA/Chapter10/googletest
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=train
TEST_TARGET=test_blob test_helper test_loss test_layer test_mnist

GTEST_SRC_INCLUDE = $(GOOGLETEST_PATH)/googletest
GMOCK_SRC_INCLUDE = $(GOOGLETEST_PATH)/googlemock
GTEST_INCLUDE = $(GOOGLETEST_PATH)/googletest/include 
GMOCK_INCLUDE = ${GOOGLETEST_PATH}/googlemock/include 

INCLUDES= -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include 
GTEST_INCLUDES = -I${GTEST_INCLUDE} -I${GTEST_SRC_INCLUDE} -I${GMOCK_INCLUDE} -I${GMOCK_SRC_INCLUDE}

NVCC_FLAGS=-m64 --resource-usage -lineinfo -lnvToolsExt

IS_CUDA_9:=$(shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2)

# Gencode arguments
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L$(CUDA_PATH)/lib64 -lcublas -lcudnn -lgomp -lcurand
ALL_CCFLAGS += -g -std=c++14 $(NVCC_FLAGS) $(INCLUDES)
GTEST_CCFLAGS += $(GTEST_INCLUDES)

# directories 
SRC_DIR = src
TEST_DIR = test
OBJ_DIR = obj
GTEST_DIR = $(GOOGLETEST_PATH)

INCS = ${SRC_DIR}/helper.h ${SRC_DIR}/blob.h ${SRC_DIR}/loss.h ${SRC_DIR}/layer.h ${SRC_DIR}/mnist.h ${SRC_DIR}/network.h

GTEST_OBJS = ${OBJ_DIR}/gtest-all.o ${OBJ_DIR}/gtest_main.o  ${OBJ_DIR}/gmock-all.o
TEST_OBJS = ${OBJ_DIR}/test_blob.o ${OBJ_DIR}/test_helper.o ${OBJ_DIR}/test_mnist.o ${OBJ_DIR}/test_loss.o ${OBJ_DIR}/test_layer.o ${OBJ_DIR}/test_network.o 

# obj of source files 
${OBJ_DIR}/loss.o: ${SRC_DIR}/loss.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/layer.o: ${SRC_DIR}/layer.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

${OBJ_DIR}/network.o: ${SRC_DIR}/network.cpp ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

${OBJ_DIR}/mnist.o: ${SRC_DIR}/mnist.cpp ${SRC_DIR}/blob.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 	

# obj of gtest
${OBJ_DIR}/gtest-all.o: ${GTEST_DIR}/googletest/src/gtest-all.cc
	@mkdir -p ${OBJ_DIR}
	$(HOST_COMPILER) -g -std=c++14  ${GTEST_CCFLAGS} -c $< -o $@

${OBJ_DIR}/gtest_main.o: ${GTEST_DIR}/googletest/src/gtest_main.cc
	@mkdir -p ${OBJ_DIR}
	$(HOST_COMPILER) -g -std=c++14  ${GTEST_CCFLAGS} -c $< -o $@

${OBJ_DIR}/gmock-all.o: ${GTEST_DIR}/googlemock/src/gmock-all.cc 
	$(HOST_COMPILER) -g -std=c++14  ${GTEST_CCFLAGS} -c $< -o $@

#objs of test 
${OBJ_DIR}/test_blob.o: ${SRC_DIR}/blob.h ${TEST_DIR}/test_blob.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) $(GTEST_CCFLAGS) -x cu -dc  ${TEST_DIR}/test_blob.cpp -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/test_helper.o: ${SRC_DIR}/helper.h ${TEST_DIR}/test_helper.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h  
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) $(GTEST_CCFLAGS) -x cu -dc ${TEST_DIR}/test_helper.cpp -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/test_loss.o: ${INCS} ${TEST_DIR}/test_loss.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) $(GTEST_CCFLAGS) -x cu -dc ${TEST_DIR}/test_loss.cpp -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/test_layer.o: ${INCS} ${TEST_DIR}/test_layer.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) $(GTEST_CCFLAGS) -x cu -dc ${TEST_DIR}/test_layer.cpp -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/test_network.o: ${INCS} ${TEST_DIR}/test_network.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) $(GTEST_CCFLAGS) -x cu -dc ${TEST_DIR}/test_network.cpp -o $@ $(GENCODE_FLAGS)		

${OBJ_DIR}/test_mnist.o: ${INCS} ${TEST_DIR}/test_mnist.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) $(GTEST_CCFLAGS) -x cu -dc ${TEST_DIR}/test_mnist.cpp -o $@ $(GENCODE_FLAGS)		

# generate test binary files 
test_blob: $(OBJ_DIR)/test_blob.o  $(GTEST_OBJS)
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

test_helper: $(OBJ_DIR)/test_helper.o $(GTEST_OBJS)
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)	

test_loss: $(OBJ_DIR)/test_loss.o $(OBJ_DIR)/loss.o $(GTEST_OBJS)
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)		

test_layer: $(OBJ_DIR)/test_layer.o $(OBJ_DIR)/layer.o  $(OBJ_DIR)/loss.o $(GTEST_OBJS) 
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

test_layer: $(OBJ_DIR)/test_layer.o $(OBJ_DIR)/loss.o  $(OBJ_DIR)/layer.o  $(GTEST_OBJS) 
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)	

test_network: $(OBJ_DIR)/test_network.o $(OBJ_DIR)/loss.o  $(OBJ_DIR)/layer.o $(OBJ_DIR)/network.o $(GTEST_OBJS) 
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)	

test_mnist: $(OBJ_DIR)/test_mnist.o $(OBJ_DIR)/mnist.o $(OBJ_DIR)/network.o $(GTEST_OBJS) 
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)	

all: test_layer test_blob test_helper test_loss test_network test_mnist
.PHONY: clean 

clean:
	rm -f ${OBJ_DIR}/*.o rm ${TEST_TARGET}

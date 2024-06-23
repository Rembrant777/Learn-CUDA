# Here we declare the path of CUDA
CUDA_PATH=/usr/local/cuda-9.2
GOOGLETEST_PATH=/home/raytrack/workspace/Learn-CUDA/Chapter10/googletest
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=train
TEST_TARGET = test_blob test_helper test_layer test_loss test_mnist test_network test_train
# test_train is integration test 

INCLUDES= -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include -I$(GOOGLETEST_PATH)/googletest/include -I$(GOOGLETEST_PATH)/googletest/
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

# directories 
SRC_DIR = src
TEST_DIR = test 
OBJ_DIR = obj
GTEST_DIR = $(GOOGLETEST_PATH)

INCS = ${SRC_DIR}/helper.h ${SRC_DIR}/blob.h ${SRC_DIR}/layer.h ${SRC_DIR}/loss.h ${SRC_DIR}/mnist.h ${SRC_DIR}/network.h
OBJS = ${OBJ_DIR}/gtest-all.o ${OBJ_DIR}/gtest_main.o ${OBJ_DIR}/mnist.o ${OBJ_DIR}/loss.o ${OBJ_DIR}/layer.o ${OBJ_DIR}/network.o

# obj of gtest
${OBJ_DIR}/gtest-all.o: ${GTEST_DIR}/googletest/src/gtest-all.cc
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/gtest_main.o: ${GTEST_DIR}/googletest/src/gtest_main.cc
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

#obj of test classes 
${OBJ_DIR}/test_blob.o: ${TEST_DIR}/test_blob.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h ${SRC_DIR}/blob.h 
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/test_helper.o: ${TEST_DIR}/test_helper.cpp ${GTEST_DIR}/googletest/include/gtest/gtest.h ${SRC_DIR}/helper.h	
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/loss.o: ${SRC_DIR}/loss.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_loss.o: ${TEST_DIR}/test_loss.cpp ${INCS} ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/layer.o: ${SRC_DIR}/layer.cu ${INCS}
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_layer.o: ${TEST_DIR}/test_layer.cpp ${INCS} ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/network.o: ${SRC_DIR}/network.cpp ${INCS}	
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_network.o: ${TEST_DIR}/test_network.cpp ${INCS} ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	

${OBJ_DIR}/mnist.o: ${SRC_DIR}/mnist.cpp ${INCS}	
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)

${OBJ_DIR}/test_mnist.o: ${TEST_DIR}/test_mnist.cpp ${INCS} ${GTEST_DIR}/googletest/include/gtest/gtest.h
	@mkdir -p ${OBJ_DIR}
	$(NVCC) $(ALL_CCFLAGS) -x cu -dc $< -o $@ $(GENCODE_FLAGS)	


# generate test binary files 
test_blob: $(OBJ_DIR)/test_blob.o $(OBJS)  
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

test_helper: $(OBJ_DIR)/test_helper.o $(OBJS) 
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

test_loss: $(OBJ_DIR)/test_loss.o $(OBJS)  
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

test_layer: $(OBJ_DIR)/test_layer.o $(OBJS)  
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)

test_network: $(OBJ_DIR)/test_network.o $(OBJS) 
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)
	
test_minist: $(OBJ_DIR)/test_minist.o $(OBJS)  
	$(NVCC) $(ALL_CCFLAGS) $^ -o $@ $(GENCODE_FLAGS) $(LIBRARIES)


all: $(OBJS)
#${TEST_TARGET}

#${TARGET} ${TEST_TARGET}
.PHONY: clean
clean:
	rm -f ${TARGET} ${OBJ_DIR}/*.o

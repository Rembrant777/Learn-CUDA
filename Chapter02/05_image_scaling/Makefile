# here we declare the path of CUDA 
CUDA_PATH=/usr/local/cuda-9.2
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=image_scaling

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_9:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 9.2}

# Gencode arguments 
SMS = 52 60 61 70 75 80
ifeq "$(IS_CUDA_9)" "1"
SMS = 52 53 60 61 62 70 72
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))


all: ${TARGET}

# scrImagePgmPpmPackage.o: scrImagePgmPpmPackage.cpp ${INCS}
# 	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# image_scaling.o: image_scaling.cu ${INCS}
# 	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# image_scaling: scrImagePgmPpmPackage.o image_scaling.o
# 	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

scrImagePgmPpmPackage.o: scrImagePgmPpmPackage.cpp scrImagePgmPpmPackage.h
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

image_scaling.o: image_scaling.cu scrImagePgmPpmPackage.h
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

${TARGET}: scrImagePgmPpmPackage.o image_scaling.o
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $+



clean:
	rm -rf ${TARGET}
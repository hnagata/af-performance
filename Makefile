CUDA_DIR = /usr/local/cuda

CXXFLAGS = -O2
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -O2 -gencode arch=compute_20,code=sm_21

INC = -I/usr/local/include -I/usr/local/cuda/include
AF_LIB = -L/usr/local/lib -lafcuda
CUDA_LIB = -L$(CUDA_DIR)/lib64 -lcuda -lcudart

all: af_sum cuda_sum

af_sum: main.cxx af_sum.cxx
	g++48 -std=c++11 -m64 -Wl,-rpath=$(CUDA_DIR)/nvvm/lib64 $(CXXFLAGS) $(INC) -o $@ $^ $(AF_LIB) $(CUDA_LIB)

cuda_sum: main.cxx cuda_sum.cu
	/usr/local/cuda/bin/nvcc -ccbin=g++48 -Xlinker -rpath=$(CUDA_DIR)/nvvm/lib64 $(NVCC_FLAGS) $(INC) -o $@ $^ $(AF_LIB) $(CUDA_LIB)

.PHONY: all

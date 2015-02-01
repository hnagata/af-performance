CUDA_DIR = /usr/local/cuda

CXXFLAGS = -std=c++11 -O2 -Wl,-rpath=$(CUDA_DIR)/nvvm/lib64
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -O2 -gencode arch=compute_20,code=sm_21 -ccbin=g++48 -Xlinker -rpath=$(CUDA_DIR)/nvvm/lib64

INC = -I/usr/local/include -I/usr/local/cuda/include
AF_LIB = -L/usr/local/lib -lafcuda
CUDA_LIB = -L$(CUDA_DIR)/lib64 -lcuda -lcudart

all: af cuda

af: af.cxx SFMT.c
	g++48 -std=c++11 -m64 $(CXXFLAGS) $(INC) -o $@ $^ $(AF_LIB) $(CUDA_LIB)

cuda: cuda.cu SFMT.c
	/usr/local/cuda/bin/nvcc $(NVCC_FLAGS) $(INC) -o $@ $^ $(AF_LIB) $(CUDA_LIB)

.PHONY: af cuda

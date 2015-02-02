#include <vector>
#include <cuda.h>
#include "perform.h"
#include "cuda_common.h"

const int BLOCK_SIZE = 256;

__global__ void sum_dev(double *a, size_t n, double *b) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int dim = blockDim.x * gridDim.x;
	extern __shared__ double tmp[];
	for (int i = idx; i < n; i += dim) {
		tmp[threadIdx.x] += a[i];
	}
	__syncthreads();

	int w = blockDim.x;
	while (w > 8) {
		w /= 2;
		if (threadIdx.x < w) tmp[threadIdx.x] += tmp[threadIdx.x + w];
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		double s = 0;
		for (int i = 0; i < w; i++) s += tmp[i];
		b[blockIdx.x] = s;
	}
}

bool prepare() {
	cudaSetDevice(0);
	return true;
}

double sum(const std::vector<double> &a) {
	int bsize = BLOCK_SIZE;
	int gsize = (a.size() + bsize - 1) / bsize;

	double *a_dev = NULL, *b_dev = NULL;
	double ans;
	CUDA_CALL(cudaMalloc((void**) &a_dev, sizeof(double) * a.size()));
	CUDA_CALL(cudaMalloc((void**) &b_dev, sizeof(double) * gsize));
	CUDA_CALL(cudaMemcpy(
			a_dev, a.data(), sizeof(double) * a.size(), 
			cudaMemcpyHostToDevice));
	sum_dev<<<gsize, bsize, bsize * sizeof(double)>>>(a_dev, a.size(), b_dev);
	CUDA_CHECK();
	sum_dev<<<1, bsize, bsize * sizeof(double)>>>(b_dev, gsize, b_dev);
	CUDA_CHECK();
	CUDA_CALL(cudaMemcpy(
			&ans, b_dev, sizeof(double), cudaMemcpyDeviceToHost));
finally:
	cudaFree(a_dev);
	cudaFree(b_dev);
	return ans;
}

#include <vector>
#include <cuda.h>
#include "perform.h"
#include "cuda_common.h"

const int GRID_SIZE = 448;
const int BLOCK_SIZE = 32;

__global__ void sum_dev(double *a, int n, double *b) {
	__shared__ double work[BLOCK_SIZE];
	const int tidx = threadIdx.x;
	const int bidx = blockIdx.x;
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int dim = blockDim.x * gridDim.x;

	work[tidx] = 0;

	for (int i = idx; i < n; i += dim) {
		work[tidx] += a[i];
	}
	__syncthreads();

	int w = BLOCK_SIZE;
	while (w > 2) {
		w /= 2;
		if (tidx < w) work[tidx] = work[tidx + w];
		__syncthreads();
	}

	if (tidx == 0) {
		b[bidx] = work[0] + work[1];
	}
}

double sum(const std::vector<double> &a) {
	double *a_dev = NULL, *b_dev = NULL;
	double *b = new double[GRID_SIZE];
	double ans = 0;	
	CUDA_CALL(cudaMalloc((void**) &a_dev, sizeof(double) * a.size()));
	CUDA_CALL(cudaMemcpy(
			a_dev, a.data(), sizeof(double) * a.size(), 
			cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc((void**) &b_dev, sizeof(double) * GRID_SIZE));
	sum_dev<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev, a.size(), b_dev);
	CUDA_CHECK();
	CUDA_CALL(cudaMemcpy(
			b, b_dev, sizeof(double) * GRID_SIZE, cudaMemcpyDeviceToHost));
	for (int i = 0; i < GRID_SIZE; i++) ans += b[i];
finally:
	cudaFree(a_dev);
	cudaFree(b_dev);
	delete[] b;
	return ans;
}

#include <vector>
#include <cuda.h>
#include "perform.h"
#include "cuda_common.h"

const int GRID_SIZE = 448;
const int BLOCK_SIZE = 32;

__global__ void sum_dev(double *a, int n, double *out) {
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
		out[bidx] = work[0] + work[1];
	}
}

double sum(const std::vector<double> &a) {
	double *a_dev = NULL;
	double ans = 0;	
	CUDA_CALL(cudaMalloc((void**) &a_dev, sizeof(double) * a.size()));
	CUDA_CALL(cudaMemcpy(
			a_dev, a.data(), sizeof(double) * a.size(), 
			cudaMemcpyHostToDevice));
	sum_dev<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev, a.size(), a_dev);
	CUDA_CHECK();
	sum_dev<<<1, BLOCK_SIZE>>>(a_dev, GRID_SIZE, a_dev);
	CUDA_CHECK();
	CUDA_CALL(cudaMemcpy(
			&ans, a_dev, sizeof(double), cudaMemcpyDeviceToHost));
finally:
	cudaFree(a_dev);
	return ans;
}

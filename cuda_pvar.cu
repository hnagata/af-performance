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

__global__ void div_dev(double *a, int b, double *out) {
	out[0] = a[0] / b;
}

__global__ void err2_dev(double *a, int n, double *mean, double *out) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int dim = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += dim) {
		double d = a[i] - mean[0];
		out[i] = d * d;
	}
}

double svar(
	cudaStream_t &stream, 
	const std::vector<double> &a, double *a_dev, int n, double *work)
{
	double out;
	CUDA_CALL(cudaMemcpyAsync(
			a_dev, a.data(), sizeof(double) * n,
			cudaMemcpyHostToDevice, stream));
	sum_dev<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a_dev, n, work);
	sum_dev<<<1, BLOCK_SIZE, 0, stream>>>(work, GRID_SIZE, work);
	div_dev<<<1, 1, 0, stream>>>(work, n, work);
	err2_dev<<<GRID_SIZE, BLOCK_SIZE>>>(a_dev, n, work, a_dev);
	sum_dev<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a_dev, n, work);
	sum_dev<<<1, BLOCK_SIZE, 0, stream>>>(work, GRID_SIZE, work);
	CUDA_CALL(cudaMemcpyAsync(
			&out, work, sizeof(double), cudaMemcpyDeviceToHost, stream));
	CUDA_CALL(cudaStreamSynchronize(stream));
finally:
	return out;
}

double pvar(const std::vector<double> &a, const std::vector<double> &b) {
	const int m = a.size(), n = b.size();
	cudaStream_t a_stream, b_stream;
	double *a_dev = NULL, *b_dev = NULL;
	double *a_work = NULL, *b_work = NULL;
	double a_svar, b_svar, ans;
	CUDA_CALL(cudaStreamCreate(&a_stream));
	CUDA_CALL(cudaStreamCreate(&b_stream));
	CUDA_CALL(cudaMalloc(
			(void**) &a_dev, sizeof(double) * (m + n + GRID_SIZE * 2)));
	b_dev = a_dev + m;
	a_work = b_dev + n;
	b_work = a_work + GRID_SIZE;

	a_svar = svar(a_stream, a, a_dev, m, a_work);
	b_svar = svar(b_stream, b, b_dev, n, b_work);
	ans = (a_svar + b_svar) / (m + n - 2);
finally:
	cudaFree(a_dev);
	cudaStreamDestroy(a_stream);
	cudaStreamDestroy(b_stream);
	return ans;
}

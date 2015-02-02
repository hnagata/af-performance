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

__global__ void err2_dev(double *a, int n, double *mean) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int dim = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += dim) {
		double d = a[i] - *mean;
		a[i] = d * d;
	}
}

double sum(double *a, int n) {
	int sum = 0;
	for (int i = 0; i < n; i++) sum += a[i];
	return sum;
}

double mean(double *a, int n) {
	return static_cast<double>(sum(a, n)) / n;
}

void sub1(
	cudaStream_t &stream, 
	const std::vector<double> &v, double *v_dev, int n,
	double *sumbuf, double *sumbuf_dev) {
	CUDA_CALL(cudaMemcpyAsync(
			v_dev, v.data(), sizeof(double) * n,
			cudaMemcpyHostToDevice, stream));
	sum_dev<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(v_dev, n, sumbuf_dev);
	CUDA_CALL(cudaMemcpyAsync(
			sumbuf, sumbuf_dev, sizeof(double) * GRID_SIZE,
			cudaMemcpyDeviceToHost, stream));
finally:
	return;
}

void sub2(
	cudaStream_t &stream, double *v_dev, int n,
	double *sumbuf, double *sumbuf_dev, double *av, double *av_dev) {
	*av = mean(sumbuf, GRID_SIZE);
	CUDA_CALL(cudaMemcpyAsync(
			av_dev, av, sizeof(double), 
			cudaMemcpyHostToDevice, stream));
	err2_dev<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(v_dev, n, av_dev);
	sum_dev<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(v_dev, n, sumbuf_dev);
	CUDA_CALL(cudaMemcpyAsync(
			sumbuf, sumbuf_dev, sizeof(double) * GRID_SIZE,
			cudaMemcpyDeviceToHost, stream));
finally:
	return;
}

typedef struct {
	double a_sumbuf[GRID_SIZE];
	double b_sumbuf[GRID_SIZE];
	double a_mean[1], b_mean[1];
} buffer_t;

double pvar(const std::vector<double> &a, const std::vector<double> &b) {
	const int m = a.size(), n = b.size();
	cudaStream_t a_stream, b_stream;
	double *a_dev = NULL, *b_dev = NULL;
	buffer_t buf, *buf_dev = NULL;
	CUDA_CALL(cudaStreamCreate(&a_stream));
	CUDA_CALL(cudaStreamCreate(&b_stream));
	CUDA_CALL(cudaMalloc((void**) &a_dev, sizeof(double) * m));
	CUDA_CALL(cudaMalloc((void**) &b_dev, sizeof(double) * n));
	CUDA_CALL(cudaMalloc((void**) &buf_dev, sizeof(buffer_t)));

	sub1(a_stream, a, a_dev, m, buf.a_sumbuf, buf_dev->a_sumbuf);
	sub1(b_stream, b, b_dev, n, buf.b_sumbuf, buf_dev->b_sumbuf);

	CUDA_CALL(cudaStreamSynchronize(a_stream));
	sub2(a_stream, a_dev, m, 
		buf.a_sumbuf, buf_dev->a_sumbuf, buf.a_mean, buf_dev->a_mean);
	CUDA_CALL(cudaStreamSynchronize(b_stream));
	sub2(b_stream, b_dev, n,
		buf.b_sumbuf, buf_dev->b_sumbuf, buf.b_mean, buf_dev->b_mean);

	CUDA_CALL(cudaStreamSynchronize(a_stream));
	buf.a_mean[0] = sum(buf.a_sumbuf, GRID_SIZE);
	CUDA_CALL(cudaStreamSynchronize(b_stream));
	buf.b_mean[0] = sum(buf.b_sumbuf, GRID_SIZE);

finally:
	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(buf_dev);
	cudaStreamDestroy(a_stream);
	cudaStreamDestroy(b_stream);

	return ((m - 1) * buf.a_mean[0] + (n - 1) * buf.b_mean[0]) / (m + n - 2);
}

bool prepare() {
	cudaSetDevice(0);
	return true;
}

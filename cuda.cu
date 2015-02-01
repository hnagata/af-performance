#include <cstdio>
#include <cuda.h>
#include <arrayfire.h>
#include "SFMT.h"

#define CUDA_CALL(s)														\
	do {																\
		cudaError_t stat = (s);											\
		if (stat != cudaSuccess) {										\
			printf("Failed in CUDA (%d): %s (ln.%d)\n",					\
				stat, __FILE__, __LINE__);								\
			goto finally;												\
		}																\
	} while (0)

#define CUDA_CHECK()							\
	do {										\
		cudaDeviceSynchronize();				\
		CUDA_CALL(cudaGetLastError());			\
	} while (0)

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

int get2fold(int x) {
	int a = 1;
	while (a < x) a <<= 1;
	return a;
}

void init_input(std::vector<double> a) {
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, 0);
	for (int i = 0; i < a.size(); i++) {
		a[i] = sfmt_genrand_real1(&sfmt) - 0.5;
	}
}

int main(int argc, char *argv[]) {
	const int N = argc <= 1 ? 1000000 : atoi(argv[1]);
	const int T = argc <= 2 ? 1000 : atoi(argv[2]);
	printf("N, T = %d, %d\n", N, T);

	std::vector<double> a(N);
	init_input(a);
	double ans = 0;

	cudaSetDevice(0);
	cudaDeviceProp prop;
	int bsize, gsize;
	cudaGetDeviceProperties(&prop, 0);
	bsize = 256; // get2fold(prop.warpSize);
	gsize = (N + bsize - 1) / bsize;
	printf("blockDim, gridDim = %d, %d\n", bsize, gsize);

	/* BEGIN */
	af::timer tm = af::timer::start();	
	double *a_dev = NULL, *b_dev = NULL;
	double tmp;
	CUDA_CALL(cudaMalloc((void**) &a_dev, sizeof(double) * N));
	CUDA_CALL(cudaMalloc((void**) &b_dev, sizeof(double) * gsize));
	CUDA_CALL(cudaMemcpy(
			a_dev, a.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
	for (int i = 0; i < T; i++) {
		sum_dev<<<gsize, bsize, bsize * sizeof(double)>>>(a_dev, N, b_dev);
		CUDA_CHECK();
		sum_dev<<<1, bsize, bsize * sizeof(double)>>>(b_dev, gsize, b_dev);
		CUDA_CHECK();
		CUDA_CALL(cudaMemcpy(
				&tmp, b_dev, sizeof(double), cudaMemcpyDeviceToHost));
		ans += tmp;
	}
finally:
	cudaFree(a_dev);
	cudaFree(b_dev);
	double t = af::timer::stop(tm);
	/* END */

	printf("value: %g\n", ans);
	printf("time: %g\n", t);
	return 0;
}

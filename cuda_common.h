#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cstdio>
#include <cuda.h>

#define CUDA_CALL(s)													\
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

#endif // CUDA_COMMON_H


#include <cuda.h>
#include "cuda_common.h"
#include "perform.h"

bool prepare() {
	bool status = false;
	CUDA_CALL(cudaSetDevice(0));
	status = true;
finally:
	return status;
}

	

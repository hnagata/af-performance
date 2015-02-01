#include <cstdio>
#include <cstdlib>
#include <arrayfire.h>
#include <cuda_runtime_api.h>
#include "SFMT.h"

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

	/* BEGIN */
	af::timer tm = af::timer::start();
	af::array a_dev(N, a.data());
	for (int i = 0; i < T; i++) {
		ans += af::sum<double>(a_dev);
	}
	double t = af::timer::stop(tm);
	/* END */

	printf("value: %g\n", ans);
	printf("time: %g\n", t);
	return 0;
}

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <arrayfire.h>
#include "perform.h"

void init_input(std::vector<double> a) {
	srand(0xcafebabe);
	for (int i = 0; i < a.size(); i++) {
		a[i] = static_cast<float>(rand()) / RAND_MAX - 0.5;
	}
}

int main(int argc, char *argv[]) {
	const int N = argc <= 1 ? 1000000 : atoi(argv[1]);
	const int T = argc <= 2 ? 1000 : atoi(argv[2]);
	printf("N = %d\n", N);
	printf("T = %d\n", T);

	std::vector<double> a(N), ans(T);
	init_input(a);
	double ans_sum = 0;

	if (!prepare()) {
		printf("Failed to prepare.");
		return 0;
	}

	af::timer tm = af::timer::start();
	for (int i = 0; i < T; i++) {
		ans[i] = pvar(a, a);
	}
	double t = af::timer::stop(tm);

	printf("value: %g\n", ans[0]);
	printf("time: %g\n", t);
	return 0;
}

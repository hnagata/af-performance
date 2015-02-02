#include <vector>
#include <arrayfire.h>
#include "perform.h"

bool prepare() {
	af::deviceset(0);
	return true;
}

double pvar(const std::vector<double> &a, const std::vector<double> &b) {
	const int m = a.size(), n = b.size();
	af::array a_dev(m, a.data());
	af::array b_dev(n, b.data());
	double a_mean = af::sum<double>(a_dev) / m;
	double b_mean = af::sum<double>(b_dev) / n;
	return 
		((m - 1) * af::sum<double>(af::pow(a_dev - a_mean, 2)) +
			(n - 1) * af::sum<double>(af::pow(b_dev - b_mean, 2))) /
		(m + n - 2);
	// return ((m - 1) * af::var(a_dev) + (n - 1) * af::var(b_dev)) / (m + n - 2);
}

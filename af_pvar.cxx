#include <vector>
#include <arrayfire.h>
#include "perform.h"

double pvar(const std::vector<double> &a, const std::vector<double> &b) {
	const int m = a.size(), n = b.size();
	af::array a_dev(m, a.data());
	af::array b_dev(n, b.data());
	double a_mean = af::sum<double>(a_dev) / m;
	double b_mean = af::sum<double>(b_dev) / n;
	double a_sqsum = af::sum<double>(af::pow(a_dev - a_mean, 2));
	double b_sqsum = af::sum<double>(af::pow(b_dev - b_mean, 2));
	return (a_sqsum + b_sqsum) / (m + n - 2);
}

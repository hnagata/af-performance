#include <vector>
#include <arrayfire.h>
#include "perform.h"

bool prepare() {
	af::deviceset(0);
	return true;
}

double sum(const std::vector<double> &a) {
	af::array a_dev(a.size(), a.data());
	return af::sum<double>(a_dev);
}

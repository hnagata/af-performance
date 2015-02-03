#include <arrayfire.h>
#include "perform.h"

bool prepare() {
	af::deviceset(0);
	return true;
}

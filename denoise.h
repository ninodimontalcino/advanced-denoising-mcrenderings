#ifndef DENOISE_H
#define DENOISE_H

#include "flt.hpp"

#define DEBUG 0

// Signature of the denoise algorithm
typedef void(*denoise_func)(scalar* ,scalar* , scalar*, scalar* , scalar*, int, int, int);
void add_function(denoise_func f, std::string name, int flop);

#endif //DENOISE_H
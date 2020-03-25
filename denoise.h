#ifndef DENOISE_H
#define DENOISE_H

#include "flt.h"

// Signature of the denoise algorithm
typedef void(*denoise_func)(buffer *, buffer *, buffer*, buffer *, buffer*, int);

#endif //DENOISE_H
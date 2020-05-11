#ifndef MEMORY_MGMT_H
#define MEMORY_MGMT_H

#include <stdio.h>
#include <stdlib.h>
#include "flt.hpp"

/* ------------------------------------------------------------------
 * MEMORY MANAGEMENT
 * -> Allocation / Freeing
 * ToDo: Cache Alignment
 * -----------------------------------------------------------------*/



void allocate_buffer(scalar **buf, int W, int H);
void allocate_buffer_zero(scalar **buf, int W, int H);

void allocate_channel(scalar **channel, int W, int H);
void allocate_channel_zero(scalar **channel, int W, int H);

void free_buffer(scalar **buf);
void free_channel(scalar **channel);

#endif 
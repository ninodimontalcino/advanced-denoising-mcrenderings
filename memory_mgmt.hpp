#ifndef MEMORY_MGMT_H
#define MEMORY_MGMT_H

#include <stdio.h>
#include <stdlib.h>
#include "flt.hpp"
#include "fltopcount.hpp"
/* ------------------------------------------------------------------
 * MEMORY MANAGEMENT
 * -> Allocation / Freeing
 * ToDo: Cache Alignment
 * -----------------------------------------------------------------*/

// MEMORY ALLOCATION
void allocate_buffer_weights_sets(bufferweightset *buf, int img_width, int img_height, int nsets, int maxr);
void allocate_buffer_weights(bufferweight *buf, int img_width, int img_height, int maxr);
void allocate_buffer(buffer *buf, int img_width, int img_height);
void allocate_channel(channel *buf, int img_width, int img_height);

// MEMORY FREEING
void free_buffer_weights_sets(bufferweightset *buf, int img_width, int img_height, int nsets, int rmax);
void free_buffer_weights(bufferweight *buf, int img_width, int img_height, int rmax);
void free_buffer(buffer *buf, int img_width);
void free_channel(channel *buf, int img_width);

#endif 
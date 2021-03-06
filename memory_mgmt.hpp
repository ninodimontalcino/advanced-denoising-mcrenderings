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

// MEMORY ALLOCATION
void allocate_buffer(buffer *buf, int img_width, int img_height);
void allocate_buffer_zero(buffer *buf, int img_width, int img_height);
void allocate_channel(channel *buf, int img_width, int img_height);
void allocate_channel_zero(channel *buf, int img_width, int img_height);

// MEMORY FREEING
void free_buffer(buffer *buf, int img_width);
void free_channel(channel *buf, int img_width);

#endif 
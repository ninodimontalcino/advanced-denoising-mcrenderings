#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include "memory_mgmt.hpp"
#include "flt.hpp"


void allocate_buffer(scalar **buf, int W, int H) {
    *buf = (scalar*) malloc(3 * W * H * sizeof(scalar));
}

void allocate_buffer_zero(scalar **buf, int W, int H) {
    *buf = (scalar*) calloc(3 * W * H, sizeof(scalar));
}

void allocate_channel(scalar **channel, int W, int H) {
    *channel = (scalar*) malloc(W * H * sizeof(scalar));
}

void allocate_channel_zero(scalar **channel, int W, int H) {
    *channel = (scalar*) calloc(W * H, sizeof(scalar));
}

void free_buffer(scalar **buf) {
    free(*buf);
}

void free_channel(scalar **channel) {
    free(*channel);
}



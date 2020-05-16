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

void allocate_buffer_aligned(scalar **buf, int W, int H) {

    *buf = static_cast<scalar *>(aligned_alloc(32, 3 * W * H * sizeof(scalar))); 
    for (int i = 0; i < 3; i ++) {
        for (int x = 0; x < W; x ++) {
            for (int y = 0; y < H; y ++) {
                (*buf)[i * W * H + x * W + y] = 0.f;
            }
        }
    }
}

void allocate_buffer_aligned_zero(scalar **buf, int W, int H) {

    *buf = static_cast<scalar *>(aligned_alloc(32, 3 * W * H * sizeof(scalar))); 
    for (int i = 0; i < 3; i ++) {
        for (int x = 0; x < W; x ++) {
            for (int y = 0; y < H; y ++) {
                (*buf)[i * W * H + x * W + y] = 0.f;
            }
        }
    }
}


void allocate_channel(scalar **channel, int W, int H) {
    *channel = (scalar*) malloc(W * H * sizeof(scalar));
}

void allocate_channel_zero(scalar **channel, int W, int H) {
    *channel = (scalar*) calloc(W * H, sizeof(scalar));
}

void allocate_channel_aligned(scalar **channel, int W, int H) {
    *channel = static_cast<scalar *>(aligned_alloc(32, W * H * sizeof(scalar)));
    for (int x = 0; x < W; x ++) {
        for (int y = 0; y < H; y ++) {
            (*channel)[x * W + y] = 0.f;
        }
    }
    
}

void allocate_channel_aligned_zero(scalar **channel, int W, int H) {
    *channel = static_cast<scalar *>(aligned_alloc(32, W * H * sizeof(scalar)));
    for (int x = 0; x < W; x ++) {
        for (int y = 0; y < H; y ++) {
            (*channel)[x * W + y] = 0.f;
        }
    }
    
}

void free_buffer(scalar **buf) {
    free(*buf);
}

void free_channel(scalar **channel) {
    free(*channel);
}



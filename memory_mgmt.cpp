#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include "memory_mgmt.hpp"
#include "flt.hpp"


void allocate_buffer(buffer *buf, int img_width, int img_height) {
    *buf = (buffer) malloc(3*sizeof(void*));
    for(int i=0;i<3;++i) {
        (*buf)[i] = (channel) malloc(img_width*sizeof(void*));
        for(int x=0;x<img_width;++x) {
            (*buf)[i][x] = (scalar*)malloc(img_height*sizeof(scalar));
        } 
    }
}

void allocate_buffer_zero(buffer *buf, int img_width, int img_height) {
    *buf = (buffer) malloc(3*sizeof(void*));
    for(int i=0;i<3;++i) {
        (*buf)[i] = (channel) malloc(img_width*sizeof(void*));
        for(int x=0;x<img_width;++x) {
            (*buf)[i][x] = (scalar*) calloc(img_height, sizeof(scalar));
            //(*buf)[i][x] = (scalar*)malloc(img_height*sizeof(scalar));
            //memset((*buf)[i][x], 0, img_height*sizeof(scalar));
        } 
    }
}

void allocate_channel(channel *buf, int img_width, int img_height) {
    *buf = (channel) malloc(img_width*sizeof(void*));
    for(int i=0;i<img_width;++i) {
        (*buf)[i] = (scalar*) malloc(img_height*sizeof(void*));
    }
}

void allocate_channel_zero(channel *buf, int img_width, int img_height) {
    *buf = (channel) malloc(img_width*sizeof(void*));
    for(int i=0;i<img_width;++i) {
        (*buf)[i] = (scalar*) calloc(img_height, sizeof(scalar));
        //(*buf)[i] = (scalar*)malloc(img_height*sizeof(scalar));
        //memset((*buf)[i], 0, img_height*sizeof(scalar));
    }
}

void free_buffer(buffer *buf, int img_width) {
    for(int i=0;i<3;++i) {
        for(int x=0;x<img_width;++x)
            free((*buf)[i][x]);
        free((*buf)[i]);
    }
    free(*buf);
}

void free_channel(channel *buf, int img_width) {
    for(int x=0;x<img_width;++x)
        free((*buf)[x]);
    free(*buf);
}


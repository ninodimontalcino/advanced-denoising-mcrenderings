#include <stdio.h>
#include <stdlib.h>

#include "memory_mgmt.hpp"
#include "flt.hpp"

void allocate_buffer_weights(bufferweightset *buf, int img_width, int img_height) {

    *buf = (bufferweightset) malloc(6*sizeof(void*));

    for(int i=0;i<6;++i) {
        (*buf)[i] = (bufferweight)malloc(img_width*sizeof(void*));

        for(int x=0;x<img_width;++x) {
            (*buf)[i][x] = (buffer)malloc(img_height*sizeof(void*));
         
            for(int y=0;y<img_height;++y) {
                (*buf)[i][x][y] = (channel)malloc(img_width*sizeof(void *));
            
                for(int x1=0;x1<img_width;++x1) {
                    (*buf)[i][x][y][x1] = (scalar*)malloc(img_height*sizeof(scalar));
                } 
            }
        }
    }
}

void allocate_buffer(buffer *buf, int img_width, int img_height) {
    *buf = (buffer) malloc(3*sizeof(void*));
    for(int i=0;i<3;++i) {
        (*buf)[i] = (channel)malloc(img_width*sizeof(void*));
        for(int x=0;x<img_width;++x) {
            (*buf)[i][x] = (scalar*)malloc(img_height*sizeof(scalar));
        } 
    }
}

void allocate_channel(channel *buf, int img_width, int img_height) {
    *buf = (channel) malloc(img_width*sizeof(void*));
    for(int i=0;i<img_width;++i) {
        (*buf)[i] = (scalar*)malloc(img_height*sizeof(void*));
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


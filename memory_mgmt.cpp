#include <stdio.h>
#include <stdlib.h>

#include "memory_mgmt.hpp"
#include "flt.hpp"
#include "fltopcount.hpp"

void allocate_buffer_weights_sets(bufferweightset *buf, int img_width, int img_height, int nsets, int maxr) {

    *buf = (bufferweightset) malloc(nsets*sizeof(void*));

    for(int i=0;i<nsets;++i) {
        (*buf)[i] = (bufferweight)malloc(img_width*sizeof(void*));

        for(int x=0;x<img_width;++x) {
            (*buf)[i][x] = (buffer)malloc(img_height*sizeof(void*));
         
            for(int y=0;y<img_height;++y) {
                (*buf)[i][x][y] = (channel)malloc((2*maxr+1)*sizeof(void *));
            
                for(int x1=0;x1<2*maxr+1;++x1) {
                    (*buf)[i][x][y][x1] = (scalar*)malloc((2*maxr+1)*sizeof(scalar));
                } 
            }
        }
    }
}

void allocate_buffer_weights(bufferweight *buf, int img_width, int img_height, int maxr) {
    (*buf) = (bufferweight)malloc(img_width*sizeof(void*));

    for(int x=0;x<img_width;++x) {
        (*buf)[x] = (buffer)malloc(img_height*sizeof(void*));
        
        for(int y=0;y<img_height;++y) {
            (*buf)[x][y] = (channel)malloc((2*maxr+1)*sizeof(void *));
        
            for(int x1=0;x1<2*maxr+1;++x1) {
                (*buf)[x][y][x1] = (scalar*)malloc((2*maxr+1)*sizeof(scalar));
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
void free_buffer_weights_sets(bufferweightset *buf, int img_width, int img_height, int nsets, int rmax) {
    for(int i=0;i<nsets;++i) {
        for(int x=0;x<img_width;++x) {
            for (int y=0; y<img_height; ++y) {
                for (int x1 = 0; x1 < 2*rmax+1; ++ x1) {
                    free((*buf)[i][x][y][x1]);
                }
                free((*buf)[i][x][y]);
            }
            free((*buf)[i][x]);
        }
        free((*buf)[i]);
    }
    free(*buf);
}

void free_buffer_weights(bufferweight *buf, int img_width, int img_height, int rmax) {
    for(int x=0;x<img_width;++x) {
        for (int y=0; y<img_height; ++y) {
            for (int x1 = 0; x1 < 2*rmax+1; ++ x1) {
                free((*buf)[x][y][x1]);
            }
            free((*buf)[x][y]);
        }
        free((*buf)[x]);
    }
    free((*buf));
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


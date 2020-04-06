#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
extern "C" {
    #include "flt.h"
}
#include "tests.hpp"

void allocate_buffer(buffer *buf) {
    *buf = (buffer) malloc(3*sizeof(void*));
    for(int i=0;i<3;++i) {
        (*buf)[i] = (channel)malloc(IMG_W*sizeof(void*));
        for(int x=0;x<IMG_W;++x) {
            (*buf)[i][x] = (scalar*)malloc(IMG_H*sizeof(scalar));
        } 
    }
}

void allocate_channel(channel *buf) {
    *buf = (channel) malloc(IMG_W*sizeof(void*));
    for(int i=0;i<IMG_W;++i) {
        (*buf)[i] = (scalar*)malloc(IMG_H*sizeof(void*));
    }
}

void free_buffer(buffer *buf) {
    for(int i=0;i<3;++i) {
        for(int x=0;x<IMG_W;++x)
            free((*buf)[i][x]);
        free((*buf)[i]);
    }
    free(*buf);
}

void free_channel(channel *buf) {
    for(int x=0;x<IMG_W;++x)
        free((*buf)[x]);
    free(*buf);
}

void load_buffer_from_txt(buffer *buf, std::string filename) {
    std::cout << "\t\tLoading file " << filename << std::endl;
    std::ifstream file;
    file.open(filename);

    allocate_buffer(buf);

    for(int y=0; y<IMG_H; ++y) {
        for(int x=0; x<IMG_W; ++x) {
            for(int i=0;i<3;++i) 
                file >> (*buf)[i][x][y];
        }
    }

    file.close();
}

void load_channel_from_txt(channel *buf, std::string filename) {
    std::cout << "\t\tLoading file " << filename << std::endl;
    std::ifstream file;
    file.open(filename);

    allocate_channel(buf);

    for(int y=0; y<IMG_H; ++y) {
        for(int x=0; x<IMG_W; ++x) {
            file >> (*buf)[x][y];
        }
    }

    file.close();
}

bool compare_scalar(scalar x, scalar y) {
    return fabs(x - y) < FLOAT_TOLERANCE;
}

bool compare_buffers(buffer buf1, buffer buf2) {
    for(int i=0;i<3;++i) {
        for(int x=0;x<IMG_W;++x) {
            for(int y=0;y<IMG_H;++y) {
                if(!compare_scalar(buf1[i][x][y], buf2[i][x][y])) {
                    std::cout << "\t\tFloats in position " << x << " " << y << " are not the same!" << std::endl;
                    std::cout << "\t\t" << buf1[i][x][y] << " instead of " << buf2[i][x][y] << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_flt() {
    std::string base_file(TEST_FILES);
    buffer c, c_var;
    channel *features = (channel*)malloc(NB_FEATURES*sizeof(void*));
    channel *f_var = (channel*)malloc(NB_FEATURES*sizeof(void*));

    buffer out, out_d, out_ref;

    std::cout << "\t- Loading input buffers" << std::endl;

    load_buffer_from_txt(&c, base_file + ".txt");
    load_buffer_from_txt(&c_var, base_file + "_variance.txt");
    load_channel_from_txt(features, base_file + "_albedo.txt");
    load_channel_from_txt(&features[1], base_file + "_depth.txt");
    load_channel_from_txt(&features[2], base_file + "_normal.txt");
    load_channel_from_txt(&f_var[0], base_file + "_albedo_variance.txt");
    load_channel_from_txt(&f_var[1], base_file + "_depth_variance.txt");
    load_channel_from_txt(&f_var[2], base_file + "_normal_variance.txt");
    
    allocate_buffer(&out);
    allocate_buffer(&out_d);

    Flt_parameters p;
    p.f = TEST_F; p.r = TEST_R; p.kc = TEST_KC; p.kf = TEST_KF; p.tau = TEST_TAU;

    std::cout << "\t- Launching function" << std::endl;

    flt(out, out_d, c, c, c_var, features, f_var, p);

    std::cout << "\t- Comparing results" << std::endl;
    load_buffer_from_txt(&out_ref, base_file + "_fltoutput.txt");
    bool compare = compare_buffers(out, out_ref);

    std::cout << "\t- Freeing buffers" << std::endl;
    free_buffer(&c);
    free_buffer(&c_var);
    free_buffer(&out);
    free_buffer(&out_d);
    free_buffer(&out_ref);
    for(int i=0;i<NB_FEATURES;++i) {
        free_channel(&features[i]);
        free_channel(&f_var[i]);
    }

    free(features);
    free(f_var);

    std::cout << "\t- FLT comparison has ended" << std::endl;   
    return compare; 
}

int main() {
    std::cout << "#### FLT Test ####" << std::endl;
    bool valid = test_flt();
    if(!valid)
        std::cout << "ERROR: FLT function not valid" << std::endl;
    return 0;
}
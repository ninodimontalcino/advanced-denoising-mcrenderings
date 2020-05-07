#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "denoise.h"
#include "validation.hpp"


scalar rmse(buffer denoised, buffer gt, int img_width, int img_height){

    scalar rmse = 0;
    scalar diff;
    scalar n = img_height * img_width * 3.f;

    // Summing up squared error term
    for (int i = 0; i < 3; i++){
        for (int x = 0; x < img_width; x++){
            for (int y = 0; y < img_height; y++){
                diff = gt[i][x][y] - denoised[i][x][y];
                rmse += diff * diff;
            }
        }
    }

    // Compute mean => MSE
    rmse = rmse / n;

    // Compute square-root => RMSE
    rmse = sqrtf(rmse);

    return rmse;

}


bool compare_scalar(scalar x, scalar y) {
    return fabs(x - y) < FLOAT_TOLERANCE;
}

bool compare_buffers(buffer buf1, buffer buf2, int img_width, int img_height) {
    for(int i=0;i<3;++i) {
        for(int x = 0; x < img_width; ++x) {
            for(int y = 0; y < img_height; ++y) {
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

double squared_diff(buffer buf1, buffer buf2, int img_width, int img_height){
    double sqr_diff = 0.f;
    
    for(int i=0;i<3;++i) {
        for(int x = 0; x < img_width; ++x) {
            for(int y = 0; y < img_height; ++y) {
                sqr_diff += (buf1[i][x][y] - buf2[i][x][y]) * (buf1[i][x][y] - buf2[i][x][y]);
            }
        }
    }

    return sqr_diff;
}


void maxAbsError(double res[4], buffer buf1, buffer buf2, int img_width, int img_height){
    double maxError = 0.f;
    int locC = 0;
    int locX = 0;
    int locY = 0;
    
    for(int i=0;i<3;++i) {
        for(int x = 0; x < img_width; ++x) {
            for(int y = 0; y < img_height; ++y) {
                double local_error = abs(buf1[i][x][y] - buf2[i][x][y]);
                if (local_error > 1e-3 && DEBUG){
                    printf("Difference of %f at position: [%d][%d][%d] \n", local_error, i, x, y);
                }
                if (local_error > maxError){
                    maxError = local_error;
                    locC = i;
                    locX = x;
                    locY = y;
                }
            }
        }
    }

    res[0] = maxError;
    res[1] = locC;
    res[2] = locX;
    res[3] = locY;
}



#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "denoise.h"
#include "validation.hpp"


scalar rmse(scalar* denoised, scalar* gt, int W, int H){

    int WH = W * H;
    scalar rmse = 0;
    scalar diff;
    scalar n = W * H * 3.f;

    // Summing up squared error term
    for (int i = 0; i < 3; i++){
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y++){
                diff = gt[i * WH + x * W +y] - denoised[i * WH + x * W +y];
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

scalar rmse_woborder(scalar* denoised, scalar* gt, int W, int H, int border){

    int WH = W * H;
    scalar rmse = 0;
    scalar diff;
    scalar n = (W - 2 * border) * (H - 2 * border) * 3.f;

    // Summing up squared error term
    for (int i = 0; i < 3; i++){
        for (int x = border; x < W - border; x++){
            for (int y = border; y < H - border; y++){
                diff = gt[i * WH + x * W +y] - denoised[i * WH + x * W +y];
                rmse += diff * diff;
                if (denoised[i * WH + x * W +y] != denoised[i * WH + x * W +y]) std::cout << i << " " << x << " " << y << "\n";
                if (isnan(gt[i * WH + x * W +y])) std::cout << i << " " << x << " " << y << "\n";
                if (isnan(-denoised[i * WH + x * W +y])) std::cout << i << " " << x << " " << y << "\n";
                if (isnan(rmse)) std::cout <<  i << " " << x << " " << y << "\n";
                //std::cout << denoised[i * WH + x * W +y] << " " << i << " " << x << " " << y << "\n";
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

bool compare_buffers(scalar* buf1, scalar* buf2, int W, int H) {

    int WH = W * H;

    for(int i=0;i<3;++i) {
        for(int x = 0; x < W; ++x) {
            for(int y = 0; y < H; ++y) {
                if(!compare_scalar(buf1[i * WH + x * W + y], buf2[i * WH + x * W + y])) {
                    std::cout << "\t\tFloats in position " << x << " " << y << " are not the same!" << std::endl;
                    std::cout << "\t\t" << buf1[i * WH + x * W + y] << " instead of " << buf2[i * WH + x * W + y] << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

double squared_diff(buffer buf1, buffer buf2, int W, int H){

    int WH = W*H;
    double sqr_diff = 0.f;
    
    for(int i=0;i<3;++i) {
        for(int x = 0; x < W; ++x) {
            for(int y = 0; y < H; ++y) {
                sqr_diff += (buf1[i * WH + x * W + y] - buf2[i * WH + x * W + y]) * (buf1[i * WH + x * W + y] - buf2[i * WH + x * W + y]);
            }
        }
    }

    return sqr_diff;
}


void maxAbsError(double res[4], scalar* buf1, scalar* buf2, int W, int H){

    int WH = W * H;

    double maxError = 0.f;
    int locC = 0;
    int locX = 0;
    int locY = 0;
    
    for(int i=0;i<3;++i) {
        for(int x = 0; x < W; ++x) {
            for(int y = 0; y < H; ++y) {
                double local_error = abs(buf1[i * WH + x * W + y] - buf2[i * WH + x * W + y]);
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



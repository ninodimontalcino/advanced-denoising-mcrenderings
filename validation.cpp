#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "validation.hpp"


scalar rmse(buffer denoised, buffer gt, int img_width, int img_height){

    double rmse = 0;
    double diff;
    double n = img_height * img_width * 3;

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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fltopcount.hpp"
#include "flt.hpp"
#include "memory_mgmt.hpp"
#include <iostream>


void flt_buffer_opcount(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters* allparams, int config, int img_width, int img_height, bufferweightset weights){

    scalar sum_weights, wc;
    Flt_parameters p = allparams[config];

    // Handling Border Cases (border section)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
            }
        }
        for (int yp = p.r+p.f ; yp < img_height - p.r+p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
            }
        }

    }
    
    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width - p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height - p.r-p.f; ++yp) {

            // Init output to 0 => TODO: maybe we can do this with calloc
            for(int i = 0; i < 3; ++i)
                output[i][xp][yp] = 0.f;

            sum_weights = 0;
            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    // Compute color Weight
                    wc = access_weight(weights, xp, yp, xq, yq, config);
                    sum_weights += wc;

                    // Add contribution term
                    for(int i=0;i<3;++i)
                        output[i][xp][yp] += input[i][xq][yq] * wc;
                }
            }
            // Normalization step
            for(int i=0;i<3;++i)
                output[i][xp][yp] /= (sum_weights + EPSILON);
        }
    }
    


}

void precompute_colors_pref(bufferweightset allweights, scalar* allsums, buffer u, buffer var_u, int img_width, int img_height, Flt_parameters p) {
    // compute colors weights for prefiltering in allweights[0] and allsums[0] for all_params[0]

    scalar sum_weights, wc;
    sum_weights = 0;
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            sum_weights = 0.f;
            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight and store
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;
                    allweights[0][xp][yp][xq][yq] = wc;
                }
            }
        }
    }
    // store sum of weights
    //allsums[0] = sum_weights;
}

void precompute_weights(bufferweightset allweights, scalar* allsums, buffer u, buffer var_u,  buffer f, buffer var_f, int img_width, int img_height, Flt_parameters* all_params) {
    // compute all color weights (allweights) for all configuration of parameters (all_params)



}

scalar access_weight(bufferweightset weights, int xp, int yp, int xq, int yq, int config) {
    return weights[config][xp][yp][xq][yq];
}
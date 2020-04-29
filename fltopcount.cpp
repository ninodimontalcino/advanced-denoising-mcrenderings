#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fltopcount.hpp"
#include "flt.hpp"
#include "memory_mgmt.hpp"
#include <iostream>


void flt_buffer_opcount(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height, bufferweightset weights){

    scalar sum_weights, wc;

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
                    wc = access_weight(weights, xp, yp, xq-xp+p.r, yq-yp+p.r, 0); // prefiltering so only 1 set of weights
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



void flt_opcount(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int config, int img_width, int img_height, bufferweightset weights) {
    scalar wc, wf, w;
    scalar sum_weights;


    // For edges, just copy in output the input
      for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                out[i][xp][yp] = input[i][xp][yp];
                out[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][xp][img_height - yp - 1] = 0.f;
            }
        }
        for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                out[i][xp][yp] = input[i][xp][yp];
                out[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][img_width - xp - 1][yp] = 0.f;
            }
        }

    }

    // Real computation
    sum_weights = 0;
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {
            
            sum_weights = 0.f;
            
            for(int i=0;i<3;++i)
                out[i][xp][yp] = 0; 

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    

                    w = access_weight(weights, xp, yp, xq-xp+p.r, yq-yp+p.r, config);
                    sum_weights += w;

                    for(int i=0;i<3;++i){
                         out[i][xp][yp] += input[i][xq][yq] * w;
                    }

                }
            }

            for(int i=0;i<3;++i){
                out[i][xp][yp] /= (sum_weights + EPSILON);

                // ToDo: Fix derivatives => Use formula from paper
                d_out_d_in[i][xp][yp] = 0.f;
            }
        }
    }
}


void flt_channel_opcount(channel output, channel input, buffer u, buffer var_u, Flt_parameters p, int config, int img_width, int img_height, bufferweightset weights){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < img_width; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            output[xp][yp] = input[xp][yp];
            output[xp][img_height - yp - 1] = input[xp][img_height - yp - 1];
        }
    }
    for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
        for(int xp = 0; xp < p.r+p.f; xp++){
            output[xp][yp] = input[xp][yp];
            output[img_width - xp - 1][yp] = input[img_width - xp - 1][yp];
        }
    }


    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            sum_weights = 0.f;

            // Init output to 0 => TODO: maybe we can do this with calloc
            output[xp][yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = access_weight(weights, xp, yp, xq-xp+p.r, yq-yp+p.r, config);
                    sum_weights += wc;

                    // Add contribution term
                    output[xp][yp] += input[xq][yq] * wc;
                }
            }

            // Normalization step
            output[xp][yp] /= (sum_weights + EPSILON);
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
                    //std::cerr << xp << " " << yp << " " << xq << " " << yq << "\n";
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;
                    allweights[0][xp][yp][xq-xp+p.r][yq-yp+p.r] = wc; // translate in range (2r+1)(2r+1)
                }
            }
        }
    }
    // store sum of weights
    allsums[0] = sum_weights;
}

void precompute_weights(bufferweightset allweights, scalar* allsums, buffer u, buffer var_u,  buffer f, buffer var_f, int img_width, int img_height, Flt_parameters* all_params) {
    // compute all color weights (allweights) for all configuration of parameters (all_params)
    scalar wc, wf;
    buffer gradients;
    Flt_parameters p;

    // FIRST candidate
    p = all_params[0];

    if(f != NULL) {
        allocate_buffer(&gradients, img_width, img_height);
        for(int i=0; i<NB_FEATURES;++i) {
            compute_gradient(gradients[i], f[i], p.r+p.f, img_width, img_height);
        }
    }

    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {
                        
            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);

                    if(f != NULL)
                        wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    else
                        wf = wc;

                    allweights[0][xp][yp][xq-xp+p.r][yq-yp+p.r] = fmin(wc, wf);
                }
            }
        }
    }

    // SECOND candidate
    p = all_params[1];
    
    if(f != NULL) {
        allocate_buffer(&gradients, img_width, img_height);
        for(int i=0; i<NB_FEATURES;++i) {
            compute_gradient(gradients[i], f[i], p.r+p.f, img_width, img_height);
        }
    }

    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {
                        
            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);

                    if(f != NULL)
                        wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    else
                        wf = wc;

                    allweights[1][xp][yp][xq-xp+p.r][yq-yp+p.r] = fmin(wc, wf);
                }
            }
        }
    }

    // THIRD candidate 
    p = all_params[2];
    
    if(f != NULL) {
        allocate_buffer(&gradients, img_width, img_height);
        for(int i=0; i<NB_FEATURES;++i) {
            compute_gradient(gradients[i], f[i], p.r+p.f, img_width, img_height);
        }
    }

    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {
                        
            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);

                    if(f != NULL)
                        wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    else
                        wf = wc;

                    allweights[2][xp][yp][xq-xp+p.r][yq-yp+p.r] = fmin(wc, wf);
                }
            }
        }
    }

    // Filter error estimates
    p = all_params[3];
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    allweights[3][xp][yp][xq-xp+p.r][yq-yp+p.r] = wc;
                }
            }
        }
    }

    // Filter Selection Maps
    p = all_params[4];
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    allweights[4][xp][yp][xq-xp+p.r][yq-yp+p.r] = wc;
                }
            }
        }
    }

    // Free memory
    if(f != NULL) {
        free_buffer(&gradients, img_width);
    }

}

scalar access_weight(bufferweightset weights, int xp, int yp, int xq, int yq, int config) {
    //if (xp==119 && yp == 119) std::cout << xp << " " << yp << " " << xq << " " << yq << "\n";
    return weights[config][xp][yp][xq][yq];
}
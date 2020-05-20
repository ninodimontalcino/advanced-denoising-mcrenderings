#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H){
    
    scalar d_r, d_g, d_b, v;
    
    for (int i = 0; i < 3; i++){ 
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y++){

                // Calculate terms
                d_r = cand_r[i][x][y] - c[i][x][y];
                d_r *= d_r;
                d_g = cand_g[i][x][y] - c[i][x][y];
                d_g *= d_g;
                d_b = cand_b[i][x][y] - c[i][x][y];
                d_b *= d_b;
                v = c_var[i][x][y];
                v *= v;

                
                // Store sure error estimate
                sure[0][x][y] += d_r - v;
                sure[1][x][y] += d_g - v;
                sure[2][x][y] += d_b - v;
            }
        }
    }
}


void filtering_basic(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W*H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W*H*sizeof(scalar));
    temp2 = (scalar*) malloc(W*H*sizeof(scalar));


    // Precompute size of neighbourhood
    scalar neigh_inv = 1./ (3*(2*p.f+1)*(2*p.f+1));

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = p.r; xp < W - p.r; ++xp) {
                    for(int yp = p.r; yp < H - p.r; ++yp) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;
                     
                        scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(c_var[i][xp][yp] + c_var[i][xq][yq]);
                        temp[xp * W + yp] += (sqdist - var_cancel) / normalization;
                    }
                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp[xp * W + yp+k];
                    }
                    temp2[xp * W + yp] = sum;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[(xp+k) * W + yp];
                    }
                    scalar weight = exp(-fmax(0.f, (sum * neigh_inv)));
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * input[i][xq][yq];
                    }
                }
            }

        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f ; yp < H - p.r - p.f; ++yp) {     
            scalar w = weight_sum[xp * W + yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][H - yp - 1] = input[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
             for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][W - xp - 1][yp] = input[i][W - xp - 1][yp];
            }
        }

    }

    free(weight_sum);
    free(temp);
    free(temp2); 
}


void feature_prefiltering(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W*H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W*H*sizeof(scalar));
    temp2 = (scalar*) malloc(W*H*sizeof(scalar));

    // Precompute size of neighbourhood
    scalar neigh_inv = 1. / (3*(2*p.f+1)*(2*p.f+1));

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){
                for(int xp = p.r; xp < W - p.r; ++xp) {
                    for(int yp = p.r; yp < H - p.r; ++yp) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;
                        
                        scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(features_var[i][xp][yp] + features_var[i][xq][yq]);
                        temp[xp * W + yp] += (sqdist - var_cancel) / normalization;
                    }
                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=1) {
                    
                    scalar sum_r0 = 0.f;

                    for (int k=-p.f; k<=p.f; k++){
                        sum_r0 += temp[xp * W + yp+k];
                    }

                    temp2[xp * W + yp] = sum_r0;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[(xp+k) * W + yp];
                    }
                    scalar weight = exp(-fmax(0.f, (sum * neigh_inv)));
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * features[i][xq][yq];
                        output_var[i][xp][yp] += weight * features_var[i][xq][yq];
                    }
                }
            }
        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {
        
            scalar w = weight_sum[xp * W + yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
                output_var[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = features[i][xp][yp];
                output[i][xp][H - yp - 1] = features[i][xp][H - yp - 1];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][xp][H - yp - 1] = features_var[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
             for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[i][xp][yp] = features[i][xp][yp];
                output[i][W - xp - 1][yp] = features[i][W - xp - 1][yp];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][W - xp - 1][yp] = features_var[i][W - xp - 1][yp];
            }
        }

    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2); 

}

void candidate_filtering(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    scalar k_f_squared = p.kf * p.kf;
    int R = p.r;
    int f = p.f;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W * H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W * H * sizeof(scalar));
    temp2 = (scalar*) malloc(W * H * sizeof(scalar));

    // Init feature weights channel
    scalar* feature_weights;
    feature_weights = (scalar*) malloc(W * H * sizeof(scalar));

    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+f; x < W - R - f; ++x) {
            for(int y =  R+f; y < H -  R - f; ++y) {
                scalar diffL = features[i][x][y] - features[i][x-1][y];
                scalar diffR = features[i][x][y] - features[i][x+1][y];
                scalar diffU = features[i][x][y] - features[i][x][y-1];
                scalar diffD = features[i][x][y] - features[i][x][y+1];
                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar neigh_inv = 1. / (3*(2*p.f+1)*(2*p.f+1));


    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = p.r; xp < W - p.r; ++xp) {
                    for(int yp = p.r; yp < H - p.r; ++yp) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        scalar distance = 0;
                                          
                        scalar sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        scalar var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                        scalar normalization = EPSILON + k_c_squared*(var_term);
                        scalar dist_var = var_cancel - sqdist;
                        temp[xp * W + yp] += (dist_var / normalization);
                    }
                }
            }

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                    for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        scalar df = feature_weights[xp * W + yp];

                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        scalar dist_var = var_cancel - sqdist;

                        scalar var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        scalar normalization = k_f_squared*fmax(p.tau, var_max);

                        df = fmin(df, (dist_var)/normalization);
                        feature_weights[xp * W + yp] = df;
                    }
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp[xp * W + yp + k];
                    }
                    temp2[xp * W + yp] = sum;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[(xp+k) * W + yp];
                    }
                    scalar color_weight = (sum * neigh_inv);
                    
                    scalar weight = exp(fmin(color_weight, feature_weights[xp * W + yp]));
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }
        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {
        
            scalar w = weight_sum[xp * W + yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
            }
        }
    }

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r + p.f; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
            for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
            }
        }
    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2);
    free(feature_weights);
    free(gradients);
}

void candidate_filtering_THIRD(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    scalar k_f_squared = p.kf * p.kf;
    int R = p.r;
    int f = p.f;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W * H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W * H * sizeof(scalar));
    temp2 = (scalar*) malloc(W * H * sizeof(scalar));

    // Init feature weights channel
    scalar* feature_weights;
    feature_weights = (scalar*) malloc(W * H * sizeof(scalar));

    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+f; x < W - R - f; ++x) {
            for(int y =  R+f; y < H -  R - f; ++y) {
                scalar diffL = features[i][x][y] - features[i][x-1][y];
                scalar diffR = features[i][x][y] - features[i][x+1][y];
                scalar diffU = features[i][x][y] - features[i][x][y-1];
                scalar diffD = features[i][x][y] - features[i][x][y+1];
                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar neigh_inv = 1. / (3*(2*p.f+1)*(2*p.f+1));


    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                    for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        scalar df = feature_weights[xp * W + yp];

                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        scalar dist_var = var_cancel - sqdist;

                        scalar var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        scalar normalization = k_f_squared*fmax(p.tau, var_max);

                        df = fmin(df, (dist_var)/normalization);
                        feature_weights[xp * W + yp] = df;
                    }
                }
            }


            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight = exp(feature_weights[xp * W + yp]);
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }
        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {
        
            scalar w = weight_sum[xp * W + yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
            }
        }
    }

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r + p.f; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
            for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
            }
        }
    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2);
    free(feature_weights);
    free(gradients);
}



void candidate_filtering_all(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int W, int H){

    int WH = W*H;

    // Get parameters
    int f_r = p[0].f;
    int f_g = p[1].f;
    int f_b = p[2].f;
    scalar tau_r = p[0].tau;
    scalar tau_g = p[1].tau;
    scalar tau_b = p[2].tau;
    scalar k_c_squared_r = p[0].kc * p[0].kc;
    scalar k_f_squared_r = p[0].kf * p[0].kf;
    scalar k_c_squared_g = p[1].kc * p[1].kc;
    scalar k_f_squared_g = p[1].kf * p[1].kf;
    scalar k_f_squared_b = p[2].kf * p[2].kf;


    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(f_r, fmax(f_g, f_b));
    int f_min = fmin(f_r, fmin(f_g, f_b));
    int R = p[0].r;


    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum_r;
    scalar* weight_sum_g;
    scalar* weight_sum_b;

    weight_sum_r = (scalar*) calloc(W * H, sizeof(scalar));
    weight_sum_g = (scalar*) calloc(W * H, sizeof(scalar));
    weight_sum_b = (scalar*) calloc(W * H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2_r;
    scalar* temp2_g;
    temp = (scalar*) malloc(W * H * sizeof(scalar));
    temp2_r = (scalar*) malloc(W * H * sizeof(scalar));
    temp2_g = (scalar*) malloc(W * H * sizeof(scalar));

    // Allocate feature weights buffer
    scalar* features_weights_r;
    scalar* features_weights_b;
    features_weights_r = (scalar*) malloc(W * H * sizeof(scalar));
    features_weights_b = (scalar*) malloc(W * H * sizeof(scalar));


    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+f_min; x < W - R - f_min; ++x) {
            for(int y =  R+f_min; y < H -  R - f_min; ++y) {
                scalar diffL = features[i][x][y] - features[i][x-1][y];
                scalar diffR = features[i][x][y] - features[i][x+1][y];
                scalar diffU = features[i][x][y] - features[i][x][y-1];
                scalar diffD = features[i][x][y] - features[i][x][y+1];

                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            
            } 
        }
    }

    // Precompute size of neighbourhood
    scalar neigh_r_inv = 1. / (3*(2*f_r+1)*(2*f_r+1));
    scalar neigh_g_inv = 1. / (3*(2*f_g+1)*(2*f_g+1));

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;   
                    
                    scalar sqdist = color[i][xp][yp] - color[i][xq][yq];
                    sqdist *= sqdist;
                    scalar var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                    scalar var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                    scalar normalization_r = EPSILON + k_c_squared_r*(var_term);
                    scalar dist_var = var_cancel - sqdist;
                    temp[xp * W + yp] += (dist_var / normalization_r);

                    }
                }
            }

            // Precompute feature weights
            // @Comment from Nino: Old loop order is faster, but this one is easier for vectorization => Still room for improvements
            memset(features_weights_r, 0, W*H*sizeof(scalar));
            memset(features_weights_b, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + f_min; xp < W - R - f_min; ++xp) {
                    for(int yp = R + f_min; yp < H - R - f_min; ++yp) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        scalar df_r = features_weights_r[xp * W + yp];
                        scalar df_b = features_weights_b[xp * W + yp];

                        
                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        scalar dist_var = var_cancel - sqdist;

                        scalar var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        scalar normalization_r = k_f_squared_r*fmax(tau_r, var_max);
                        scalar normalization_b = k_f_squared_b*fmax(tau_b, var_max);

                        df_r = fmin(df_r, (dist_var)/normalization_r);
                        df_b = fmin(df_b, (dist_var)/normalization_b);
                        
                        features_weights_r[xp * W + yp] = df_r;
                        features_weights_b[xp * W + yp] = df_b;
                    } 
                }
            }
            

            // Next Steps: Box-Filtering for Patch Contribution 
            // => Use Box-Filter Seperability => linear scans of data
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_r; yp < H - R - f_r; ++yp) {
                    scalar sum_r = 0.f;
                    for (int k=-f_r; k<=f_r; k++){
                        sum_r += temp[xp * W + yp+k];
                    }
                    temp2_r[xp * W + yp] = sum_r;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
                for(int yp = R + f_r; yp < H - R - f_r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k=-f_r; k<=f_r; k++){
                        sum += temp2_r[(xp+k) * W + yp];
                    }
                    scalar color_weight = (sum * neigh_r_inv);

                    // Compute final weight
                    scalar weight = exp(fmin(color_weight, features_weights_r[xp * W + yp]));
                    weight_sum_r[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; ++yp) {
                    scalar sum_g = 0.f;
                    for (int k=-f_g; k<=f_g; k++){
                        sum_g += temp[xp * W + yp+k];
                    }
                    temp2_g[xp * W + yp] = sum_g;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k=-f_g; k<=f_g; k++){
                        sum += temp2_g[(xp+k) * W + yp];
                    }
                    scalar color_weight = (sum * neigh_g_inv);
                    
                    // Compute final weight
                    scalar weight = exp(fmin(color_weight, features_weights_r[xp * W + yp]));
                    weight_sum_g[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
                for(int yp = R + f_b; yp < H - R - f_b; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight = exp(features_weights_b[xp * W + yp]);
                    weight_sum_b[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < H - R - f_r; ++yp) {
        
            scalar w = weight_sum_r[xp * W + yp];
            for (int i=0; i<3; i++){
                output_r[i][xp][yp] /= w;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < H - R - f_g; ++yp) {
        
            scalar w = weight_sum_g[xp * W + yp];
            for (int i=0; i<3; i++){
                output_g[i][xp][yp] /= w;
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < H - R - f_b; ++yp) {
        
            scalar w = weight_sum_b[xp * W + yp];
            for (int i=0; i<3; i++){
                output_b[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + f_r; yp++){
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_r; xp++){
            for (int yp = R + f_r ; yp < H - R - f_r; yp++){
            
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
             }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + f_g; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_g; xp++){
            for (int yp = R + f_g ; yp < H - R - f_g; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
            }
        }
    }

    // Free memory
    free(weight_sum_r);
    free(weight_sum_g);
    free(weight_sum_b);
    free(temp2_r);
    free(temp2_g);
    free(features_weights_r);
    free(features_weights_b);
    free(gradients);

}

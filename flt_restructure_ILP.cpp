#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all_ILP(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H){
    
    scalar d_r, d_g, d_b, v;
    
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){

            scalar sure_r = 0.f;
            scalar sure_g = 0.f;
            scalar sure_b = 0.f;

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Calculate terms
                d_r = cand_r[i][x][y] - c[i][x][y];
                d_r *= d_r;
                d_g = cand_g[i][x][y] - c[i][x][y];
                d_g *= d_g;
                d_b = cand_b[i][x][y] - c[i][x][y];
                d_b *= d_b;
                v = c_var[i][x][y];
                v *= v;

                // Summing up
                sure_r += d_r - v; 
                sure_g += d_g - v; 
                sure_b += d_b - v; 

            }
            // Store sure error estimate
            sure[0][x][y] = sure_r;
            sure[1][x][y] = sure_g;
            sure[2][x][y] = sure_b;
        }
    }
}


void filtering_basic_ILP(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    

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
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r; yp < H - p.r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance = 0;
                    for (int i=0; i<3; i++){                        
                        scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(c_var[i][xp][yp] + c_var[i][xq][yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp * W + yp] = distance;

                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=8) {

                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
                    scalar sum_4 = 0.f;
                    scalar sum_5 = 0.f;
                    scalar sum_6 = 0.f;
                    scalar sum_7 = 0.f;

                    for (int k=-p.f; k<=p.f; k++){
                        sum_0 += temp[xp * W + yp+k];
                        sum_1 += temp[xp * W + yp+k+1];
                        sum_2 += temp[xp * W + yp+k+2];
                        sum_3 += temp[xp * W + yp+k+3];
                        sum_4 += temp[xp * W + yp+k+4];
                        sum_5 += temp[xp * W + yp+k+5];
                        sum_6 += temp[xp * W + yp+k+6];
                        sum_7 += temp[xp * W + yp+k+7];
                    }

                    temp2[xp * W + yp] = sum_0;
                    temp2[xp * W + yp+1] = sum_1;
                    temp2[xp * W + yp+2] = sum_2;
                    temp2[xp * W + yp+3] = sum_3;
                    temp2[xp * W + yp+4] = sum_4;
                    temp2[xp * W + yp+5] = sum_5;
                    temp2[xp * W + yp+6] = sum_6;
                    temp2[xp * W + yp+7] = sum_7;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    for (int k=-p.f; k<=p.f; k++){
                        sum_0 += temp2[(xp+k) * W + yp];
                        sum_1 += temp2[(xp+k) * W + yp+1];
                        sum_2 += temp2[(xp+k) * W + yp+2];
                        sum_3 += temp2[(xp+k) * W + yp+3];
                    }
                    
                    scalar weight_0 = exp(-fmax(0.f, (sum_0 * neigh_inv)));
                    scalar weight_1 = exp(-fmax(0.f, (sum_1 * neigh_inv)));
                    scalar weight_2 = exp(-fmax(0.f, (sum_2 * neigh_inv)));
                    scalar weight_3 = exp(-fmax(0.f, (sum_3 * neigh_inv)));
                    
                    weight_sum[xp * W + yp] += weight_0;
                    weight_sum[xp * W + yp+1] += weight_1;
                    weight_sum[xp * W + yp+2] += weight_2;
                    weight_sum[xp * W + yp+3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight_0 * input[i][xq][yq];
                        output[i][xp][yp+1] += weight_1 * input[i][xq][yq+1];
                        output[i][xp][yp+2] += weight_2 * input[i][xq][yq+2];
                        output[i][xp][yp+3] += weight_3 * input[i][xq][yq+3];
                    }
                }
            }

        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f ; yp < H - p.r - p.f; yp+=4) {     
            
            scalar w_0 = weight_sum[xp * W + yp];
            scalar w_1 = weight_sum[xp * W + yp+1];
            scalar w_2 = weight_sum[xp * W + yp+2];
            scalar w_3 = weight_sum[xp * W + yp+3];

            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w_0;
                output[i][xp][yp+1] /= w_1;
                output[i][xp][yp+2] /= w_2;
                output[i][xp][yp+3] /= w_3;
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


void feature_prefiltering_ILP(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

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
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r; yp < H - p.r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance = 0.f;

                    for (int i=0; i<3; i++){                        
                        scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(features_var[i][xp][yp] + features_var[i][xq][yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp * W + yp] = distance;

                }
            }
            

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=8) {
                    
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
                    scalar sum_4 = 0.f;
                    scalar sum_5 = 0.f;
                    scalar sum_6 = 0.f;
                    scalar sum_7 = 0.f;

                    for (int k=-p.f; k<=p.f; k++){
                        sum_0 += temp[xp * W + yp+k];
                        sum_1 += temp[xp * W + yp+k+1];
                        sum_2 += temp[xp * W + yp+k+2];
                        sum_3 += temp[xp * W + yp+k+3];
                        sum_4 += temp[xp * W + yp+k+4];
                        sum_5 += temp[xp * W + yp+k+5];
                        sum_6 += temp[xp * W + yp+k+6];
                        sum_7 += temp[xp * W + yp+k+7];
                    }
                    temp2[xp * W + yp] = sum_0;
                    temp2[xp * W + yp+1] = sum_1;
                    temp2[xp * W + yp+2] = sum_2;
                    temp2[xp * W + yp+3] = sum_3;
                    temp2[xp * W + yp+4] = sum_4;
                    temp2[xp * W + yp+5] = sum_5;
                    temp2[xp * W + yp+6] = sum_6;
                    temp2[xp * W + yp+7] = sum_7;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    for (int k=-p.f; k<=p.f; k++){
                        sum_0 += temp2[(xp+k) * W + yp];
                        sum_1 += temp2[(xp+k) * W + yp+1];
                        sum_2 += temp2[(xp+k) * W + yp+2];
                        sum_3 += temp2[(xp+k) * W + yp+3];
                    }

                    scalar weight_0 = exp(-fmax(0.f, (sum_0 * neigh_inv)));
                    scalar weight_1 = exp(-fmax(0.f, (sum_1 * neigh_inv)));
                    scalar weight_2 = exp(-fmax(0.f, (sum_2 * neigh_inv)));
                    scalar weight_3 = exp(-fmax(0.f, (sum_3 * neigh_inv)));

                    weight_sum[xp * W + yp] += weight_0;
                    weight_sum[xp * W + yp+1] += weight_1;
                    weight_sum[xp * W + yp+2] += weight_2;
                    weight_sum[xp * W + yp+3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight_0 * features[i][xq][yq];
                        output[i][xp][yp+1] += weight_1 * features[i][xq][yq+1];
                        output[i][xp][yp+2] += weight_2 * features[i][xq][yq+2];
                        output[i][xp][yp+3] += weight_3 * features[i][xq][yq+3];
                        output_var[i][xp][yp] += weight_0 * features_var[i][xq][yq];
                        output_var[i][xp][yp+1] += weight_1 * features_var[i][xq][yq+1];
                        output_var[i][xp][yp+2] += weight_2 * features_var[i][xq][yq+2];
                        output_var[i][xp][yp+3] += weight_3 * features_var[i][xq][yq+3];
                    }
                }
            }


        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=4) {
        
            scalar w_0 = weight_sum[xp * W + yp];
            scalar w_1 = weight_sum[xp * W + yp+1];
            scalar w_2 = weight_sum[xp * W + yp+2];
            scalar w_3 = weight_sum[xp * W + yp+3];

            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w_0;
                output_var[i][xp][yp] /= w_0;
                output[i][xp][yp+1] /= w_1;
                output_var[i][xp][yp+1] /= w_1;
                output[i][xp][yp+2] /= w_2;
                output_var[i][xp][yp+2] /= w_2;
                output[i][xp][yp+3] /= w_3;
                output_var[i][xp][yp+3] /= w_3;
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

void candidate_filtering_all_ILP(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int W, int H){

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


    // Precompute normalization constants
    /*
    scalar* norm_r;
    scalar* norm_b;
    norm_r = (scalar*) malloc(3 * W * H * sizeof(scalar));
    norm_b = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int j=0; j<NB_FEATURES; ++j) {
        for(int x =  R+f_min; x < W - R - f_min; ++x) {
            for(int y =  R+f_min; y < H -  R - f_min; ++y) {

                scalar diffL = features[j][x][y] - features[j][x-1][y];
                scalar diffR = features[j][x][y] - features[j][x+1][y];
                scalar diffU = features[j][x][y] - features[j][x][y-1];
                scalar diffD = features[j][x][y] - features[j][x][y+1];

                scalar max_r = fmax(features_var[j][x][y], tau_r);
                scalar max_b = fmax(features_var[j][x][y], tau_b);
                scalar gradient = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
                norm_r[j * WH + x * W + y] = k_f_squared_r*fmax(max_r, gradient);
                norm_b[j * WH + x * W + y] = k_f_squared_b*fmax(max_b, gradient);
            }
        } 
    }
    */
    
    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int x =  R+f_min; x < W - R - f_min; ++x) {
        for(int y =  R+f_min; y < H -  R - f_min; ++y) {
            for(int i=0; i<NB_FEATURES;++i) {
                scalar diffL = features[i][x][y] - features[i][x-1][y];
                scalar diffR = features[i][x][y] - features[i][x+1][y];
                scalar diffU = features[i][x][y] - features[i][x][y-1];
                scalar diffD = features[i][x][y] - features[i][x][y+1];

                gradients[3 * (x * W + y) + i] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar neigh_r_inv = 1. / (3*(2*f_r+1)*(2*f_r+1));
    scalar neigh_g_inv = 1. / (3*(2*f_g+1)*(2*f_g+1));

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // #######################################################################################
            // WEIGHT COMPUTATION
            // #######################################################################################

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
            for(int xp = R + f_min; xp < W - R - f_min; ++xp) {
                for(int yp = R + f_min; yp < H - R - f_min; ++yp) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r = 0.f;
                    scalar df_b = 0.f;

                    for(int j=0; j<NB_FEATURES;++j){
                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        scalar dist_var = var_cancel - sqdist;

                        // ============ !!!!!!! =================================================================
                        // ToDo: Precompute normalization constants => always the same independet of R
                        // @Comment from Nino: Not successfull so far => same runtime but less cycles => yet less performance
                        scalar var_max = fmax(features_var[j][xp][yp], gradients[3 * (xp * W + yp) + j]);
                        scalar normalization_r = k_f_squared_r*fmax(tau_r, var_max);
                        scalar normalization_b = k_f_squared_b*fmax(tau_b, var_max);
                        // ============ !!!!!!! =================================================================

                        //df_r = fmin(df_r, dist_var/norm_r[j * WH + xp * W + yp]);
                        //df_b = fmin(df_b, dist_var/norm_b[j * WH + xp * W + yp]);
                        df_r = fmin(df_r, (dist_var)/normalization_r);
                        df_b = fmin(df_b, (dist_var)/normalization_b);
                    }

                    features_weights_r[xp * W + yp] = df_r;
                    features_weights_b[xp * W + yp] = df_b;
                } 
            }

            // #######################################################################################
            // BOX FILTERING => seperability of box filter kernel => two linear operations
            // #######################################################################################
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_r; yp < H - R - f_r; yp+=8) {
                    scalar sum_r_0 = 0.f;
                    scalar sum_r_1 = 0.f;
                    scalar sum_r_2 = 0.f;
                    scalar sum_r_3 = 0.f;
                    scalar sum_r_4 = 0.f;
                    scalar sum_r_5 = 0.f;
                    scalar sum_r_6 = 0.f;
                    scalar sum_r_7 = 0.f;

                    for (int k=-f_r; k<=f_r; k++){
                        sum_r_0 += temp[xp * W + yp+k];
                        sum_r_1 += temp[xp * W + yp+k+1];
                        sum_r_2 += temp[xp * W + yp+k+2];
                        sum_r_3 += temp[xp * W + yp+k+3];
                        sum_r_4 += temp[xp * W + yp+k+4];
                        sum_r_5 += temp[xp * W + yp+k+5];
                        sum_r_6 += temp[xp * W + yp+k+6];
                        sum_r_7 += temp[xp * W + yp+k+7];
                    }
                    temp2_r[xp * W + yp] = sum_r_0;
                    temp2_r[xp * W + yp+1] = sum_r_1;
                    temp2_r[xp * W + yp+2] = sum_r_2;
                    temp2_r[xp * W + yp+3] = sum_r_3;
                    temp2_r[xp * W + yp+4] = sum_r_4;
                    temp2_r[xp * W + yp+5] = sum_r_5;
                    temp2_r[xp * W + yp+6] = sum_r_6;
                    temp2_r[xp * W + yp+7] = sum_r_7;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
                for(int yp = R + f_r; yp < H - R - f_r; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    for (int k=-f_r; k<=f_r; k++){
                        sum_0 += temp2_r[(xp+k) * W + yp];
                        sum_1 += temp2_r[(xp+k) * W + yp+1];
                        sum_2 += temp2_r[(xp+k) * W + yp+2];
                        sum_3 += temp2_r[(xp+k) * W + yp+3];
                    }
                    scalar color_weight_0 = (sum_0 * neigh_r_inv);
                    scalar color_weight_1 = (sum_1 * neigh_r_inv);
                    scalar color_weight_2 = (sum_2 * neigh_r_inv);
                    scalar color_weight_3 = (sum_3 * neigh_r_inv);

                    // Compute final weight
                    scalar weight_0 = exp(fmin(color_weight_0, features_weights_r[xp * W + yp]));
                    scalar weight_1 = exp(fmin(color_weight_1, features_weights_r[xp * W + yp+1]));
                    scalar weight_2 = exp(fmin(color_weight_2, features_weights_r[xp * W + yp+2]));
                    scalar weight_3 = exp(fmin(color_weight_3, features_weights_r[xp * W + yp+3]));

                    weight_sum_r[xp * W + yp] += weight_0;
                    weight_sum_r[xp * W + yp+1] += weight_1;
                    weight_sum_r[xp * W + yp+2] += weight_2;
                    weight_sum_r[xp * W + yp+3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_r[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_r[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_r[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; yp+=8) {
                    
                    scalar sum_g_0 = 0.f;
                    scalar sum_g_1 = 0.f;
                    scalar sum_g_2 = 0.f;
                    scalar sum_g_3 = 0.f;
                    scalar sum_g_4 = 0.f;
                    scalar sum_g_5 = 0.f;
                    scalar sum_g_6 = 0.f;
                    scalar sum_g_7 = 0.f;

                    for (int k=-f_g; k<=f_g; k++){
                        sum_g_0 += temp[xp * W + yp+k];
                        sum_g_1 += temp[xp * W + yp+k+1];
                        sum_g_2 += temp[xp * W + yp+k+2];
                        sum_g_3 += temp[xp * W + yp+k+3];
                        sum_g_4 += temp[xp * W + yp+k+4];
                        sum_g_5 += temp[xp * W + yp+k+5];
                        sum_g_6 += temp[xp * W + yp+k+6];
                        sum_g_7 += temp[xp * W + yp+k+7];
                    }
                    temp2_g[xp * W + yp] = sum_g_0;
                    temp2_g[xp * W + yp+1] = sum_g_1;
                    temp2_g[xp * W + yp+2] = sum_g_2;
                    temp2_g[xp * W + yp+3] = sum_g_3;
                    temp2_g[xp * W + yp+4] = sum_g_4;
                    temp2_g[xp * W + yp+5] = sum_g_5;
                    temp2_g[xp * W + yp+6] = sum_g_6;
                    temp2_g[xp * W + yp+7] = sum_g_7;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    for (int k=-f_g; k<=f_g; k++){
                        sum_0 += temp2_g[(xp+k) * W + yp];
                        sum_1 += temp2_g[(xp+k) * W + yp+1];
                        sum_2 += temp2_g[(xp+k) * W + yp+2];
                        sum_3 += temp2_g[(xp+k) * W + yp+3];
                    }

                    scalar color_weight_0 = (sum_0 * neigh_g_inv);
                    scalar color_weight_1 = (sum_1 * neigh_g_inv);
                    scalar color_weight_2 = (sum_2 * neigh_g_inv);
                    scalar color_weight_3 = (sum_3 * neigh_g_inv);
                    
                    // Compute final weight
                    scalar weight_0 = exp(fmin(color_weight_0, features_weights_r[xp * W + yp]));
                    scalar weight_1 = exp(fmin(color_weight_1, features_weights_r[xp * W + yp+1]));
                    scalar weight_2 = exp(fmin(color_weight_2, features_weights_r[xp * W + yp+2]));
                    scalar weight_3 = exp(fmin(color_weight_3, features_weights_r[xp * W + yp+3]));

                    weight_sum_g[xp * W + yp] += weight_0;
                    weight_sum_g[xp * W + yp+1] += weight_1;
                    weight_sum_g[xp * W + yp+2] += weight_2;
                    weight_sum_g[xp * W + yp+3] += weight_3;
                    
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_g[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_g[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_g[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
                for(int yp = R + f_b; yp < H - R - f_b; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar weight_0 = exp(features_weights_b[xp * W + yp]);
                    scalar weight_1 = exp(features_weights_b[xp * W + yp+1]);
                    scalar weight_2 = exp(features_weights_b[xp * W + yp+2]);
                    scalar weight_3 = exp(features_weights_b[xp * W + yp+3]);
                    scalar weight_4 = exp(features_weights_b[xp * W + yp+4]);
                    scalar weight_5 = exp(features_weights_b[xp * W + yp+5]);
                    scalar weight_6 = exp(features_weights_b[xp * W + yp+6]);
                    scalar weight_7 = exp(features_weights_b[xp * W + yp+7]);

                    weight_sum_b[xp * W + yp] += weight_0;
                    weight_sum_b[xp * W + yp+1] += weight_1;
                    weight_sum_b[xp * W + yp+2] += weight_2;
                    weight_sum_b[xp * W + yp+3] += weight_3;
                    weight_sum_b[xp * W + yp+4] += weight_4;
                    weight_sum_b[xp * W + yp+5] += weight_5;
                    weight_sum_b[xp * W + yp+6] += weight_6;
                    weight_sum_b[xp * W + yp+7] += weight_7;

                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_b[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_b[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_b[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                        output_b[i][xp][yp+4] += weight_4 * color[i][xq][yq+4];
                        output_b[i][xp][yp+5] += weight_5 * color[i][xq][yq+5];
                        output_b[i][xp][yp+6] += weight_6 * color[i][xq][yq+6];
                        output_b[i][xp][yp+7] += weight_7 * color[i][xq][yq+7];
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < H - R - f_r; yp+=4) {
        
            scalar w_0 = weight_sum_r[xp * W + yp];
            scalar w_1 = weight_sum_r[xp * W + yp+1];
            scalar w_2 = weight_sum_r[xp * W + yp+2];
            scalar w_3 = weight_sum_r[xp * W + yp+3];

            for (int i=0; i<3; i++){
                output_r[i][xp][yp] /= w_0;
                output_r[i][xp][yp+1] /= w_1;
                output_r[i][xp][yp+2] /= w_2;
                output_r[i][xp][yp+3] /= w_3;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < H - R - f_g; yp+=4) {
        
            scalar w_0 = weight_sum_g[xp * W + yp];
            scalar w_1 = weight_sum_g[xp * W + yp+1];
            scalar w_2 = weight_sum_g[xp * W + yp+2];
            scalar w_3 = weight_sum_g[xp * W + yp+3];

            for (int i=0; i<3; i++){
                output_g[i][xp][yp] /= w_0;
                output_g[i][xp][yp+1] /= w_1;
                output_g[i][xp][yp+2] /= w_2;
                output_g[i][xp][yp+3] /= w_3;
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < H - R - f_b; yp+=4) {
        
            scalar w_0 = weight_sum_b[xp * W + yp];
            scalar w_1 = weight_sum_b[xp * W + yp+1];
            scalar w_2 = weight_sum_b[xp * W + yp+2];
            scalar w_3 = weight_sum_b[xp * W + yp+3];

            for (int i=0; i<3; i++){
                output_b[i][xp][yp] /= w_0;
                output_b[i][xp][yp+1] /= w_1;
                output_b[i][xp][yp+2] /= w_2;
                output_b[i][xp][yp+3] /= w_3;
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
    free(temp);
    free(temp2_r);
    free(&temp2_g);
    free(features_weights_r);
    free(features_weights_b);
    //free(norm_r);
    //free(norm_b);
    free(gradients);

}

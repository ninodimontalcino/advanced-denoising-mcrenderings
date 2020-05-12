#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all_ILP(scalar* sure, scalar* c, scalar* c_var, scalar* cand_r, scalar* cand_g, scalar* cand_b, int W, int H){
    
    scalar d_r, d_g, d_b, v;
    
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){

            scalar sure_r = 0.f;
            scalar sure_g = 0.f;
            scalar sure_b = 0.f;

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Calculate terms
                d_r = cand_r[3 * (x * W + y) + i] - c[3 * (x * W + y) + i];
                d_r *= d_r;
                d_g = cand_g[3 * (x * W + y) + i] - c[3 * (x * W + y) + i];
                d_g *= d_g;
                d_b = cand_b[3 * (x * W + y) + i] - c[3 * (x * W + y) + i];
                d_b *= d_b;
                v = c_var[3 * (x * W + y) + i];
                v *= v;

                // Summing up
                sure_r += d_r - v; 
                sure_g += d_g - v; 
                sure_b += d_b - v; 

            }
            // Store sure error estimate
            sure[0 + 3 * (x * W + y)] = sure_r;
            sure[1 + 3 * (x * W + y)] = sure_g;
            sure[2 + 3 * (x * W + y)] = sure_b;
        }
    }
}


void filtering_basic_ILP(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int W, int H){
    
    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    allocate_channel_zero(&weight_sum, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    allocate_channel(&temp, W, H);
    allocate_channel_zero(&temp2, W, H);


    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r; yp < H - p.r; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance_0 = 0.f;
                    scalar distance_1 = 0.f;
                    scalar distance_2 = 0.f;
                    scalar distance_3 = 0.f;
                    /*
                    scalar distance_4 = 0.f;
                    scalar distance_5 = 0.f;
                    scalar distance_6 = 0.f;
                    scalar distance_7 = 0.f;
                    */


                    for (int i=0; i<3; i++){                        
                        scalar sqdist_0 = c[3 * (xp * W + yp + 0) + i] - c[3 * (xq * W + yq + 0) + i];
                        scalar sqdist_1 = c[3 * (xp * W + yp + 1) + i] - c[3 * (xq * W + yq + 1) + i];
                        scalar sqdist_2 = c[3 * (xp * W + yp + 2) + i] - c[3 * (xq * W + yq + 2) + i];
                        scalar sqdist_3 = c[3 * (xp * W + yp + 3) + i] - c[3 * (xq * W + yq + 3) + i];
                        /*
                        scalar sqdist_4 = c[3 * (xp * W + yp + 4) + i] - c[3 * (xq * W + yq + 4) + i];
                        scalar sqdist_5 = c[3 * (xp * W + yp + 5) + i] - c[3 * (xq * W + yq + 5) + i];
                        scalar sqdist_6 = c[3 * (xp * W + yp + 6) + i] - c[3 * (xq * W + yq + 6) + i];
                        scalar sqdist_7 = c[3 * (xp * W + yp + 7) + i] - c[3 * (xq * W + yq + 7) + i];
                        scalar sqdist_8 = c[3 * (xp * W + yp + 8) + i] - c[3 * (xq * W + yq + 8) + i];
                        */

                        sqdist_0 *= sqdist_0;
                        sqdist_1 *= sqdist_1;
                        sqdist_2 *= sqdist_2;
                        sqdist_3 *= sqdist_3;
                        /*
                        sqdist_4 *= sqdist_4;
                        sqdist_5 *= sqdist_5;
                        sqdist_6 *= sqdist_6;
                        sqdist_7 *= sqdist_7;
                        */

                        scalar var_cancel_0 = c_var[3 * (xp * W + yp + 0) + i] + fmin(c_var[3 * (xp * W + yp + 0) + i], c_var[3 * (xq * W + yq + 0) + i]);
                        scalar var_cancel_1 = c_var[3 * (xp * W + yp + 1) + i] + fmin(c_var[3 * (xp * W + yp + 1) + i], c_var[3 * (xq * W + yq + 1) + i]);
                        scalar var_cancel_2 = c_var[3 * (xp * W + yp + 2) + i] + fmin(c_var[3 * (xp * W + yp + 2) + i], c_var[3 * (xq * W + yq + 2) + i]);
                        scalar var_cancel_3 = c_var[3 * (xp * W + yp + 3) + i] + fmin(c_var[3 * (xp * W + yp + 3) + i], c_var[3 * (xq * W + yq + 3) + i]);
                        /*
                        scalar var_cancel_4 = c_var[3 * (xp * W + yp + 4) + i] + fmin(c_var[3 * (xp * W + yp + 4) + i], c_var[3 * (xq * W + yq + 4) + i]);
                        scalar var_cancel_5 = c_var[3 * (xp * W + yp + 5) + i] + fmin(c_var[3 * (xp * W + yp + 5) + i], c_var[3 * (xq * W + yq + 5) + i]);
                        scalar var_cancel_6 = c_var[3 * (xp * W + yp + 6) + i] + fmin(c_var[3 * (xp * W + yp + 6) + i], c_var[3 * (xq * W + yq + 6) + i]);
                        scalar var_cancel_7 = c_var[3 * (xp * W + yp + 7) + i] + fmin(c_var[3 * (xp * W + yp + 7) + i], c_var[3 * (xq * W + yq + 7) + i]);
                        */

                        scalar normalization_0 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 0) + i] + c_var[3 * (xq * W + yq + 0) + i]);
                        scalar normalization_1 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 1) + i] + c_var[3 * (xq * W + yq + 1) + i]);
                        scalar normalization_2 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 2) + i] + c_var[3 * (xq * W + yq + 2) + i]);
                        scalar normalization_3 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 3) + i] + c_var[3 * (xq * W + yq + 3) + i]);
                        /*
                        scalar normalization_4 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 4) + i] + c_var[3 * (xq * W + yq + 4) + i]);
                        scalar normalization_5 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 5) + i] + c_var[3 * (xq * W + yq + 5) + i]);
                        scalar normalization_6 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 6) + i] + c_var[3 * (xq * W + yq + 6) + i]);
                        scalar normalization_7 = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp + 7) + i] + c_var[3 * (xq * W + yq + 7) + i]);
                        */

                        distance_0 += (sqdist_0 - var_cancel_0) / normalization_0;
                        distance_1 += (sqdist_1 - var_cancel_1) / normalization_1;
                        distance_2 += (sqdist_2 - var_cancel_2) / normalization_2;
                        distance_3 += (sqdist_3 - var_cancel_3) / normalization_3;
                        /*
                        distance_4 += (sqdist_4 - var_cancel_4) / normalization_4;
                        distance_5 += (sqdist_5 - var_cancel_5) / normalization_5;
                        distance_6 += (sqdist_6 - var_cancel_6) / normalization_6;
                        distance_7 += (sqdist_7 - var_cancel_7) / normalization_7;
                        */
                    }

                    temp[xp * W + yp + 0] = distance_0;
                    temp[xp * W + yp + 1] = distance_1;
                    temp[xp * W + yp + 2] = distance_2;
                    temp[xp * W + yp + 3] = distance_3;
                    /*
                    temp[xp * W + yp + 4] = distance_4;
                    temp[xp * W + yp + 5] = distance_5;
                    temp[xp * W + yp + 6] = distance_6;
                    temp[xp * W + yp + 7] = distance_7;
                    */


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

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[(xp+k) * W + yp];
                    }
                    scalar weight = exp(-fmax(0.f, (sum / neigh)));
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[3 * (xp * W + yp) + i] += weight * input[3 * (xq * W + yq) + i];
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
                output[3 * (xp * W + yp) + i] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            for (int i = 0; i < 3; i++){
                output[3 * (xp * W + yp) + i] = input[3 * (xp * W + yp) + i];
                output[i + 3 * (xp * W + H - yp - 1)] = input[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < p.r+p.f; xp++){
        for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
            for (int i = 0; i < 3; i++){
                output[3 * (xp * W + yp) + i] = input[3 * (xp * W + yp) + i];
                output[3 * ((W - xp - 1) * W + yp) + i] = input[3 * ((W - xp - 1) * W + yp) + i];
            }
        }
    }

    free(weight_sum);
    free(temp);
    free(temp2); 
}


void feature_prefiltering_ILP(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int W, int H){

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    allocate_channel_zero(&weight_sum, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    allocate_channel(&temp, W, H);
    allocate_channel_zero(&temp2, W, H);

    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);

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
                        scalar sqdist = features[3 * (xp * W + yp) + i] - features[3 * (xq * W + yq) + i];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[3 * (xp * W + yp) + i] + fmin(features_var[3 * (xp * W + yp) + i], features_var[3 * (xq * W + yq) + i]);
                        scalar normalization = EPSILON + k_c_squared*(features_var[3 * (xp * W + yp) + i] + features_var[3 * (xq * W + yq) + i]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp * W + yp] = distance;

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

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[(xp+k) * W + yp];
                    }
                    scalar weight = exp(-fmax(0.f, (sum / neigh)));
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[3 * (xp * W + yp) + i] += weight * features[3 * (xq * W + yq) + i];
                        output_var[3 * (xp * W + yp) + i] += weight * features_var[3 * (xq * W + yq) + i];
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
                output[3 * (xp * W + yp) + i] /= w;
                output_var[3 * (xp * W + yp) + i] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            for (int i = 0; i < 3; i++){
                output[3 * (xp * W + yp) + i] = features[3 * (xp * W + yp) + i];
                output[3 * (xp * W + H - yp - 1) + i] = features[3 * (xp * W + H - yp - 1) + i];
                output_var[3 * (xp * W + yp) + i] = features_var[3 * (xp * W + yp) + i];
                output_var[3 * (xp * W + H - yp - 1) + i] = features_var[3 * (xp * W + H - yp - 1) + i];
            }
        }
    }
    for(int xp = 0; xp < p.r+p.f; xp++){
        for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
            for (int i = 0; i < 3; i++){
                output[3 * (xp * W + yp) + i] = features[3 * (xp * W + yp) + i];
                output[3 * ((W - xp - 1) * W + yp) + i] = features[3 * ((W - xp - 1) * W + yp) + i];
                output_var[3 * (xp * W + yp) + i] = features_var[3 * (xp * W + yp) + i];
                output_var[3 * ((W - xp - 1) * W + yp) + i] = features_var[3 * ((W - xp - 1) * W + yp) + i];
            }
        }
    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2); 

}

void candidate_filtering_all_ILP(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int W, int H){

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
    scalar* weight_sum;
    allocate_buffer_zero(&weight_sum, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2_r;
    scalar* temp2_g;
    allocate_channel(&temp, W, H); 
    allocate_channel(&temp2_r, W, H); 
    allocate_channel(&temp2_g, W, H); 

    // Allocate feature weights buffer
    scalar* features_weights_r;
    scalar* features_weights_b;
    allocate_channel(&features_weights_r, W, H);
    allocate_channel(&features_weights_b, W, H);


    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int x =  R+f_min; x < W - R - f_min; ++x) {
        for(int y =  R+f_min; y < H -  R - f_min; ++y) {
                
                scalar diffL_00 = features[3 * (x * W + y) + 0] - features[3 * ((x - 1) * W + y) + 0];
                scalar diffL_01 = features[3 * (x * W + y) + 1] - features[3 * ((x - 1) * W + y) + 1];
                scalar diffL_02 = features[3 * (x * W + y) + 2] - features[3 * ((x - 1) * W + y) + 2];

                scalar diffR_00 = features[3 * (x * W + y) + 0] - features[3 * ((x + 1) * W + y) + 0];
                scalar diffR_01 = features[3 * (x * W + y) + 1] - features[3 * ((x + 1) * W + y) + 1];
                scalar diffR_02 = features[3 * (x * W + y) + 2] - features[3 * ((x + 1) * W + y) + 2];

                scalar diffU_00 = features[3 * (x * W + y) + 0] - features[3 * (x * W + y - 1) + 0];
                scalar diffU_01 = features[3 * (x * W + y) + 1] - features[3 * (x * W + y - 1) + 1];
                scalar diffU_02 = features[3 * (x * W + y) + 2] - features[3 * (x * W + y - 1) + 2];

                scalar diffD_00 = features[3 * (x * W + y) + 0] - features[3 * (x * W + y + 1) + 0];
                scalar diffD_01 = features[3 * (x * W + y) + 1] - features[3 * (x * W + y + 1) + 1];
                scalar diffD_02 = features[3 * (x * W + y) + 2] - features[3 * (x * W + y + 1) + 2];

                gradients[3 * (x * W + y) + 0] = fmin(diffL_00*diffL_00, diffR_00*diffR_00) + fmin(diffU_00*diffU_00, diffD_00*diffD_00);
                gradients[3 * (x * W + y) + 1] = fmin(diffL_01*diffL_01, diffR_01*diffR_01) + fmin(diffU_01*diffU_01, diffD_01*diffD_01);
                gradients[3 * (x * W + y) + 2] = fmin(diffL_02*diffL_02, diffR_02*diffR_02) + fmin(diffU_02*diffU_02, diffD_02*diffD_02);
        } 
    }

    // Precompute size of neighbourhood
    scalar neigh_r = 3*(2*f_r+1)*(2*f_r+1);
    scalar neigh_g = 3*(2*f_g+1)*(2*f_g+1);
    scalar neigh_b = 3*(2*f_b+1)*(2*f_b+1);

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
           for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sqdist_00 = color[3 * (xp * W + yp) + 0] - color[3 * (xq * W + yq) + 0];
                    scalar sqdist_01 = color[3 * (xp * W + yp) + 1] - color[3 * (xq * W + yq) + 1];
                    scalar sqdist_02 = color[3 * (xp * W + yp) + 2] - color[3 * (xq * W + yq) + 2];

                    sqdist_00 *= sqdist_00;
                    sqdist_01 *= sqdist_01;
                    sqdist_02 *= sqdist_02;

                    scalar var_cancel_00 = color_var[3 * (xp * W + yp) + 0] + fmin(color_var[3 * (xp * W + yp) + 0], color_var[3 * (xq * W + yq) + 0]);
                    scalar var_cancel_01 = color_var[3 * (xp * W + yp) + 1] + fmin(color_var[3 * (xp * W + yp) + 1], color_var[3 * (xq * W + yq) + 1]);
                    scalar var_cancel_02 = color_var[3 * (xp * W + yp) + 2] + fmin(color_var[3 * (xp * W + yp) + 2], color_var[3 * (xq * W + yq) + 2]);

                    scalar var_term_00 = color_var[3 * (xp * W + yp) + 0] + color_var[3 * (xq * W + yq) + 0];
                    scalar var_term_01 = color_var[3 * (xp * W + yp) + 1] + color_var[3 * (xq * W + yq) + 1];
                    scalar var_term_02 = color_var[3 * (xp * W + yp) + 2] + color_var[3 * (xq * W + yq) + 2];

                    scalar normalization_r_00 = EPSILON + k_c_squared_r*(var_term_00);
                    scalar normalization_r_01 = EPSILON + k_c_squared_r*(var_term_01);
                    scalar normalization_r_02 = EPSILON + k_c_squared_r*(var_term_02);

                    scalar dist_var_00 = sqdist_00 - var_cancel_00;
                    scalar dist_var_01 = sqdist_01 - var_cancel_01;
                    scalar dist_var_02 = sqdist_02 - var_cancel_02;

                    temp[xp * W + yp] = ((dist_var_00) / normalization_r_00) + ((dist_var_01) / normalization_r_01) + ((dist_var_02) / normalization_r_02);
                    
                }
            }

            // Precompute feature weights
            for(int xp = R + f_min; xp < W - R - f_min; ++xp) {
                for(int yp = R + f_min; yp < H - R - f_min; yp+=4) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r_0 = 0.f;
                    scalar df_r_1 = 0.f;
                    scalar df_r_2 = 0.f;
                    scalar df_r_3 = 0.f;

                    scalar df_b_0 = 0.f;
                    scalar df_b_1 = 0.f;
                    scalar df_b_2 = 0.f;
                    scalar df_b_3 = 0.f;

                    for(int j=0; j<NB_FEATURES;++j){
                        
                        scalar sqdist_0 = features[3 * (xp * W + yp) + j] - features[3 * (xq * W + yq) + j];
                        scalar sqdist_1 = features[3 * (xp * W + yp+1) + j] - features[3 * (xq * W + yq+1) + j];
                        scalar sqdist_2 = features[3 * (xp * W + yp+2) + j] - features[3 * (xq * W + yq+2) + j];
                        scalar sqdist_3 = features[3 * (xp * W + yp+3) + j] - features[3 * (xq * W + yq+3) + j];

                        sqdist_0 *= sqdist_0;
                        sqdist_1 *= sqdist_1;
                        sqdist_2 *= sqdist_2;
                        sqdist_3 *= sqdist_3;

                        scalar var_cancel_0 = features_var[3 * (xp * W + yp) + j] + fmin(features_var[3 * (xp * W + yp) + j], features_var[3 * (xq * W + yq) + j]);
                        scalar var_cancel_1 = features_var[3 * (xp * W + yp+1) + j] + fmin(features_var[3 * (xp * W + yp+1) + j], features_var[3 * (xq * W + yq+1) + j]);
                        scalar var_cancel_2 = features_var[3 * (xp * W + yp+2) + j] + fmin(features_var[3 * (xp * W + yp+2) + j], features_var[3 * (xq * W + yq+2) + j]);
                        scalar var_cancel_3 = features_var[3 * (xp * W + yp+3) + j] + fmin(features_var[3 * (xp * W + yp+3) + j], features_var[3 * (xq * W + yq+3) + j]);
                        
                        scalar var_max_0 = fmax(features_var[3 * (xp * W + yp) + j], gradients[3 * (xp * W + yp) + j]);
                        scalar var_max_1 = fmax(features_var[3 * (xp * W + yp+1) + j], gradients[3 * (xp * W + yp+1) + j]);
                        scalar var_max_2 = fmax(features_var[3 * (xp * W + yp+2) + j], gradients[3 * (xp * W + yp+2) + j]);
                        scalar var_max_3 = fmax(features_var[3 * (xp * W + yp+3) + j], gradients[3 * (xp * W + yp+3) + j]);

                        scalar normalization_r_0 = k_f_squared_r*fmax(tau_r, var_max_0);
                        scalar normalization_r_1 = k_f_squared_r*fmax(tau_r, var_max_1);
                        scalar normalization_r_2 = k_f_squared_r*fmax(tau_r, var_max_2);
                        scalar normalization_r_3 = k_f_squared_r*fmax(tau_r, var_max_3);

                        scalar normalization_b_0 = k_f_squared_b*fmax(tau_b, var_max_0);
                        scalar normalization_b_1 = k_f_squared_b*fmax(tau_b, var_max_1);
                        scalar normalization_b_2 = k_f_squared_b*fmax(tau_b, var_max_2);
                        scalar normalization_b_3 = k_f_squared_b*fmax(tau_b, var_max_3);

                        scalar dist_var_0 = sqdist_0 - var_cancel_0;
                        scalar dist_var_1 = sqdist_1 - var_cancel_1;
                        scalar dist_var_2 = sqdist_2 - var_cancel_2;
                        scalar dist_var_3 = sqdist_3 - var_cancel_3;

                        df_r_0 = fmax(df_r_0, (dist_var_0)/normalization_r_0);
                        df_r_1 = fmax(df_r_1, (dist_var_1)/normalization_r_1);
                        df_r_2 = fmax(df_r_2, (dist_var_2)/normalization_r_2);
                        df_r_3 = fmax(df_r_3, (dist_var_3)/normalization_r_3);

                        df_b_0 = fmax(df_b_0, (dist_var_0)/normalization_b_0);
                        df_b_1 = fmax(df_b_1, (dist_var_1)/normalization_b_1);
                        df_b_2 = fmax(df_b_2, (dist_var_2)/normalization_b_2);
                        df_b_3 = fmax(df_b_3, (dist_var_3)/normalization_b_3);
                    }

                    features_weights_r[xp * W + yp] = exp(-df_r_0);
                    features_weights_b[xp * W + yp] = exp(-df_b_0);
                    features_weights_r[xp * W + yp+1] = exp(-df_r_1);
                    features_weights_b[xp * W + yp+1] = exp(-df_b_1);
                    features_weights_r[xp * W + yp+2] = exp(-df_r_2);
                    features_weights_b[xp * W + yp+2] = exp(-df_b_2);
                    features_weights_r[xp * W + yp+3] = exp(-df_r_3);
                    features_weights_b[xp * W + yp+3] = exp(-df_b_3);
                } 
            }
            

            // Next Steps: Box-Filtering for Patch Contribution 
            // => Use Box-Filter Seperability => linear scans of data
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_r; yp < H - R - f_r; yp+=8) {
                    scalar sum_r0 = 0.f;
                    scalar sum_r1 = 0.f;
                    scalar sum_r2 = 0.f;
                    scalar sum_r3 = 0.f;
                    scalar sum_r4 = 0.f;
                    scalar sum_r5 = 0.f;
                    scalar sum_r6 = 0.f;
                    scalar sum_r7 = 0.f;
                     /*
                    scalar sum_r8 = 0.f;
                    scalar sum_r9 = 0.f;
                    scalar sum_r10 = 0.f;
                    scalar sum_r11 = 0.f;
                    scalar sum_r12 = 0.f;
                    scalar sum_r13 = 0.f;
                    scalar sum_r14 = 0.f;
                    scalar sum_r15 = 0.f;
                    */

                    for (int k=-f_r; k<=f_r; k++){
                        sum_r0 += temp[xp * W + yp+k];
                        sum_r1 += temp[xp * W + yp+k+1];
                        sum_r2 += temp[xp * W + yp+k+2];
                        sum_r3 += temp[xp * W + yp+k+3];
                        sum_r4 += temp[xp * W + yp+k+4];
                        sum_r5 += temp[xp * W + yp+k+5];
                        sum_r6 += temp[xp * W + yp+k+6];
                        sum_r7 += temp[xp * W + yp+k+7];
                        /*
                        sum_r8 += temp[xp * W + yp+k+8];
                        sum_r9 += temp[xp * W + yp+k+9];
                        sum_r10 += temp[xp * W + yp+k+10];
                        sum_r11 += temp[xp * W + yp+k+11];
                        sum_r12 += temp[xp * W + yp+k+12];
                        sum_r13 += temp[xp * W + yp+k+13];
                        sum_r14 += temp[xp * W + yp+k+14];
                        sum_r15 += temp[xp * W + yp+k+15];
                        */
                    }
                    temp2_r[xp * W + yp] = sum_r0;
                    temp2_r[xp * W + yp+1] = sum_r1;
                    temp2_r[xp * W + yp+2] = sum_r2;
                    temp2_r[xp * W + yp+3] = sum_r3;
                    temp2_r[xp * W + yp+4] = sum_r4;
                    temp2_r[xp * W + yp+5] = sum_r5;
                    temp2_r[xp * W + yp+6] = sum_r6;
                    temp2_r[xp * W + yp+7] = sum_r7;
                    /*
                    temp2_r[xp * W + yp+8] = sum_r8;
                    temp2_r[xp * W + yp+9] = sum_r9;
                    temp2_r[xp * W + yp+10] = sum_r10;
                    temp2_r[xp * W + yp+11] = sum_r11;
                    temp2_r[xp * W + yp+12] = sum_r12;
                    temp2_r[xp * W + yp+13] = sum_r13;
                    temp2_r[xp * W + yp+14] = sum_r14;
                    temp2_r[xp * W + yp+15] = sum_r15;
                    */
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
                        sum_0 += temp2_r[(xp+k)*W + yp];
                        sum_1 += temp2_r[(xp+k)*W + yp+1];
                        sum_2 += temp2_r[(xp+k)*W + yp+2];
                        sum_3 += temp2_r[(xp+k)*W + yp+3];
                    }

                    scalar color_weight_0 = exp(-fmax(0.f, (sum_0 / neigh_r)));
                    scalar color_weight_1 = exp(-fmax(0.f, (sum_1 / neigh_r)));
                    scalar color_weight_2 = exp(-fmax(0.f, (sum_2 / neigh_r)));
                    scalar color_weight_3 = exp(-fmax(0.f, (sum_3 / neigh_r)));

                    // Compute final weight
                    scalar weight_0 = fmin(color_weight_0, features_weights_r[xp * W + yp]);
                    scalar weight_1 = fmin(color_weight_1, features_weights_r[xp * W + yp+1]);
                    scalar weight_2 = fmin(color_weight_2, features_weights_r[xp * W + yp+2]);
                    scalar weight_3 = fmin(color_weight_3, features_weights_r[xp * W + yp+3]);
                    
                    weight_sum[3 * (xp * W + yp)] += weight_0;
                    weight_sum[3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[3 * (xp * W + yp+3)] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output_r[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_r[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_r[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_r[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];

                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; yp+=8) {
                    
                    scalar sum_g0 = 0.f;
                    scalar sum_g1 = 0.f;
                    scalar sum_g2 = 0.f;
                    scalar sum_g3 = 0.f;
                    scalar sum_g4 = 0.f;
                    scalar sum_g5 = 0.f;
                    scalar sum_g6 = 0.f;
                    scalar sum_g7 = 0.f;
                    /*
                    scalar sum_g8 = 0.f;
                    scalar sum_g9 = 0.f;
                    scalar sum_g10 = 0.f;
                    scalar sum_g11 = 0.f;
                    scalar sum_g12 = 0.f;
                    scalar sum_g13 = 0.f;
                    scalar sum_g14 = 0.f;
                    scalar sum_g15 = 0.f;
                    */

                    for (int k=-f_g; k<=f_g; k++){
                        sum_g0 += temp[xp * W + yp+k];
                        sum_g1 += temp[xp * W + yp+k+1];
                        sum_g2 += temp[xp * W + yp+k+2];
                        sum_g3 += temp[xp * W + yp+k+3];
                        sum_g4 += temp[xp * W + yp+k+4];
                        sum_g5 += temp[xp * W + yp+k+5];
                        sum_g6 += temp[xp * W + yp+k+6];
                        sum_g7 += temp[xp * W + yp+k+7];
                        /*
                        sum_g8 += temp[xp * W + yp+k+8];
                        sum_g9 += temp[xp * W + yp+k+9];
                        sum_g10 += temp[xp * W + yp+k+10];
                        sum_g11 += temp[xp * W + yp+k+11];
                        sum_g12 += temp[xp * W + yp+k+12];
                        sum_g13 += temp[xp * W + yp+k+13];
                        sum_g14 += temp[xp * W + yp+k+14];
                        sum_g15 += temp[xp * W + yp+k+15];
                        */
                    }
                    temp2_g[xp * W + yp] = sum_g0;
                    temp2_g[xp * W + yp+1] = sum_g1;
                    temp2_g[xp * W + yp+2] = sum_g2;
                    temp2_g[xp * W + yp+3] = sum_g3;
                    temp2_g[xp * W + yp+4] = sum_g4;
                    temp2_g[xp * W + yp+5] = sum_g5;
                    temp2_g[xp * W + yp+6] = sum_g6;
                    temp2_g[xp * W + yp+7] = sum_g7;
                    /*
                    temp2_g[xp * W + yp+8] = sum_g8;
                    temp2_g[xp * W + yp+9] = sum_g9;
                    temp2_g[xp * W + yp+10] = sum_g10;
                    temp2_g[xp * W + yp+11] = sum_g11;
                    temp2_g[xp * W + yp+12] = sum_g12;
                    temp2_g[xp * W + yp+13] = sum_g13;
                    temp2_g[xp * W + yp+14] = sum_g14;
                    temp2_g[xp * W + yp+15] = sum_g15;
                    */
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
                    /*
                    scalar sum_4 = 0.f;
                    scalar sum_5 = 0.f;
                    scalar sum_6 = 0.f;
                    scalar sum_7 = 0.f;
                    scalar sum_8 = 0.f;
                    scalar sum_9 = 0.f;
                    scalar sum_10 = 0.f;
                    scalar sum_11 = 0.f;
                    scalar sum_12 = 0.f;
                    scalar sum_13 = 0.f;
                    scalar sum_14 = 0.f;
                    scalar sum_15 = 0.f;
                    */

                    for (int k=-f_g; k<=f_g; k++){
                        sum_0 += temp2_g[(xp+k)*W + yp];
                        sum_1 += temp2_g[(xp+k)*W + yp+1];
                        sum_2 += temp2_g[(xp+k)*W + yp+2];
                        sum_3 += temp2_g[(xp+k)*W + yp+3];
                        /*
                        sum_4 += temp2_g[(xp+k)*W + yp+4];
                        sum_5 += temp2_g[(xp+k)*W + yp+5];
                        sum_6 += temp2_g[(xp+k)*W + yp+6];
                        sum_7 += temp2_g[(xp+k)*W + yp+7];
                        sum_8 += temp2_g[(xp+k)*W + yp+8];
                        sum_9 += temp2_g[(xp+k)*W + yp+9];
                        sum_10 += temp2_g[(xp+k)*W + yp+10];
                        sum_11 += temp2_g[(xp+k)*W + yp+11];
                        sum_12 += temp2_g[(xp+k)*W + yp+12];
                        sum_13 += temp2_g[(xp+k)*W + yp+13];
                        sum_14 += temp2_g[(xp+k)*W + yp+14];
                        sum_15 += temp2_g[(xp+k)*W + yp+15];
                        */
                    }
                    scalar color_weight_0 = exp(-fmax(0.f, (sum_0 / neigh_g)));
                    scalar color_weight_1 = exp(-fmax(0.f, (sum_1 / neigh_g)));
                    scalar color_weight_2 = exp(-fmax(0.f, (sum_2 / neigh_g)));
                    scalar color_weight_3 = exp(-fmax(0.f, (sum_3 / neigh_g)));
                    /*
                    scalar color_weight_4 = exp(-fmax(0.f, (sum_4 / neigh_g)));
                    scalar color_weight_5 = exp(-fmax(0.f, (sum_5 / neigh_g)));
                    scalar color_weight_6 = exp(-fmax(0.f, (sum_6 / neigh_g)));
                    scalar color_weight_7 = exp(-fmax(0.f, (sum_7 / neigh_g)));
                    scalar color_weight_8 = exp(-fmax(0.f, (sum_8 / neigh_g)));
                    scalar color_weight_9 = exp(-fmax(0.f, (sum_9 / neigh_g)));
                    scalar color_weight_10 = exp(-fmax(0.f, (sum_10 / neigh_g)));
                    scalar color_weight_11 = exp(-fmax(0.f, (sum_11 / neigh_g)));
                    scalar color_weight_12 = exp(-fmax(0.f, (sum_12 / neigh_g)));
                    scalar color_weight_13 = exp(-fmax(0.f, (sum_13 / neigh_g)));
                    scalar color_weight_14 = exp(-fmax(0.f, (sum_14 / neigh_g)));
                    scalar color_weight_15 = exp(-fmax(0.f, (sum_15 / neigh_g)));
                    */
                    
                    // Compute final weight
                    scalar weight_0 = fmin(color_weight_0, features_weights_r[xp * W + yp]);
                    scalar weight_1 = fmin(color_weight_1, features_weights_r[xp * W + yp+1]);
                    scalar weight_2 = fmin(color_weight_2, features_weights_r[xp * W + yp+2]);
                    scalar weight_3 = fmin(color_weight_3, features_weights_r[xp * W + yp+3]);
                    /*
                    scalar weight_4 = fmin(color_weight_4, features_weights_r[xp * W + yp+4]);
                    scalar weight_5 = fmin(color_weight_5, features_weights_r[xp * W + yp+5]);
                    scalar weight_6 = fmin(color_weight_6, features_weights_r[xp * W + yp+6]);
                    scalar weight_7 = fmin(color_weight_7, features_weights_r[xp * W + yp+7]);
                    scalar weight_8 = fmin(color_weight_8, features_weights_r[xp * W + yp+8]);
                    scalar weight_9 = fmin(color_weight_9, features_weights_r[xp * W + yp+9]);
                    scalar weight_10 = fmin(color_weight_10, features_weights_r[xp * W + yp+10]);
                    scalar weight_11 = fmin(color_weight_11, features_weights_r[xp * W + yp+11]);
                    scalar weight_12 = fmin(color_weight_12, features_weights_r[xp * W + yp+12]);
                    scalar weight_13 = fmin(color_weight_13, features_weights_r[xp * W + yp+13]);
                    scalar weight_14 = fmin(color_weight_14, features_weights_r[xp * W + yp+14]);
                    scalar weight_15 = fmin(color_weight_15, features_weights_r[xp * W + yp+15]);
                    */

                    weight_sum[1 + 3 * (xp * W + yp)] += weight_0;
                    weight_sum[1 + 3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[1 + 3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[1 + 3 * (xp * W + yp+3)] += weight_3;
                    /*
                    weight_sum[1 + 3 * (xp * W + yp+4)] += weight_4;
                    weight_sum[1 + 3 * (xp * W + yp+5)] += weight_5;
                    weight_sum[1 + 3 * (xp * W + yp+6)] += weight_6;
                    weight_sum[1 + 3 * (xp * W + yp+7)] += weight_7;
                    weight_sum[1 + 3 * (xp * W + yp+8)] += weight_8;
                    weight_sum[1 + 3 * (xp * W + yp+9)] += weight_9;
                    weight_sum[1 + 3 * (xp * W + yp+10)] += weight_10;
                    weight_sum[1 + 3 * (xp * W + yp+11)] += weight_11;
                    weight_sum[1 + 3 * (xp * W + yp+12)] += weight_12;
                    weight_sum[1 + 3 * (xp * W + yp+13)] += weight_13;
                    weight_sum[1 + 3 * (xp * W + yp+14)] += weight_14;
                    weight_sum[1 + 3 * (xp * W + yp+15)] += weight_15;
                    */
                    
                    for (int i=0; i<3; i++){
                        output_g[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_g[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_g[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_g[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];
                        /*
                        output_g[3 * (xp * W + yp+4) + i] += weight_4 * color[3 * (xq * W + yq+4) + i];
                        output_g[3 * (xp * W + yp+5) + i] += weight_5 * color[3 * (xq * W + yq+5) + i];
                        output_g[3 * (xp * W + yp+6) + i] += weight_6 * color[3 * (xq * W + yq+6) + i];
                        output_g[3 * (xp * W + yp+7) + i] += weight_7 * color[3 * (xq * W + yq+7) + i];
                        output_g[3 * (xp * W + yp+8) + i] += weight_8 * color[3 * (xq * W + yq+8) + i];
                        output_g[3 * (xp * W + yp+9) + i] += weight_9 * color[3 * (xq * W + yq+9) + i];
                        output_g[3 * (xp * W + yp+10) + i] += weight_10 * color[3 * (xq * W + yq+10) + i];
                        output_g[3 * (xp * W + yp+11) + i] += weight_11 * color[3 * (xq * W + yq+11) + i];
                        output_g[3 * (xp * W + yp+12) + i] += weight_12 * color[3 * (xq * W + yq+12) + i];
                        output_g[3 * (xp * W + yp+13) + i] += weight_13 * color[3 * (xq * W + yq+13) + i];
                        output_g[3 * (xp * W + yp+14) + i] += weight_14 * color[3 * (xq * W + yq+14) + i];
                        output_g[3 * (xp * W + yp+15) + i] += weight_15 * color[3 * (xq * W + yq+15) + i];
                        */
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
                for(int yp = R + f_b; yp < H - R - f_b; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight_0 = features_weights_b[xp * W + yp];
                    scalar weight_1 = features_weights_b[xp * W + yp+1];
                    scalar weight_2 = features_weights_b[xp * W + yp+2];
                    scalar weight_3 = features_weights_b[xp * W + yp+3];
                    /*
                    scalar weight_4 = features_weights_b[xp * W + yp+4];
                    scalar weight_5 = features_weights_b[xp * W + yp+5];
                    scalar weight_6 = features_weights_b[xp * W + yp+6];
                    scalar weight_7 = features_weights_b[xp * W + yp+7];
                    scalar weight_8 = features_weights_b[xp * W + yp+8];
                    scalar weight_9 = features_weights_b[xp * W + yp+9];
                    scalar weight_10 = features_weights_b[xp * W + yp+10];
                    scalar weight_11 = features_weights_b[xp * W + yp+11];
                    scalar weight_12 = features_weights_b[xp * W + yp+12];
                    scalar weight_13 = features_weights_b[xp * W + yp+13];
                    scalar weight_14 = features_weights_b[xp * W + yp+14];
                    scalar weight_15 = features_weights_b[xp * W + yp+15];
                    */
                    
                    weight_sum[2 + 3 * (xp * W + yp)] += weight_0;
                    weight_sum[2 + 3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[2 + 3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[2 + 3 * (xp * W + yp+3)] += weight_3;
                    /*
                    weight_sum[2 + 3 * (xp * W + yp+4)] += weight_4;
                    weight_sum[2 + 3 * (xp * W + yp+5)] += weight_5;
                    weight_sum[2 + 3 * (xp * W + yp+6)] += weight_6;
                    weight_sum[2 + 3 * (xp * W + yp+7)] += weight_7;
                    weight_sum[2 + 3 * (xp * W + yp+8)] += weight_8;
                    weight_sum[2 + 3 * (xp * W + yp+9)] += weight_9;
                    weight_sum[2 + 3 * (xp * W + yp+10)] += weight_10;
                    weight_sum[2 + 3 * (xp * W + yp+11)] += weight_11;
                    weight_sum[2 + 3 * (xp * W + yp+12)] += weight_12;
                    weight_sum[2 + 3 * (xp * W + yp+13)] += weight_13;
                    weight_sum[2 + 3 * (xp * W + yp+14)] += weight_14;
                    weight_sum[2 + 3 * (xp * W + yp+15)] += weight_15;
                    */
                    
                    for (int i=0; i<3; i++){
                        output_b[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_b[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_b[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_b[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];
                        /*
                        output_b[3 * (xp * W + yp+4) + i] += weight_4 * color[3 * (xq * W + yq+4) + i];
                        output_b[3 * (xp * W + yp+5) + i] += weight_5 * color[3 * (xq * W + yq+5) + i];
                        output_b[3 * (xp * W + yp+6) + i] += weight_6 * color[3 * (xq * W + yq+6) + i];
                        output_b[3 * (xp * W + yp+7) + i] += weight_7 * color[3 * (xq * W + yq+7) + i];
                        output_b[3 * (xp * W + yp+8) + i] += weight_8 * color[3 * (xq * W + yq+8) + i];
                        output_b[3 * (xp * W + yp+9) + i] += weight_9 * color[3 * (xq * W + yq+9) + i];
                        output_b[3 * (xp * W + yp+10) + i] += weight_10 * color[3 * (xq * W + yq+10) + i];
                        output_b[3 * (xp * W + yp+11) + i] += weight_11 * color[3 * (xq * W + yq+11) + i];
                        output_b[3 * (xp * W + yp+12) + i] += weight_12 * color[3 * (xq * W + yq+12) + i];
                        output_b[3 * (xp * W + yp+13) + i] += weight_13 * color[3 * (xq * W + yq+13) + i];
                        output_b[3 * (xp * W + yp+14) + i] += weight_14 * color[3 * (xq * W + yq+14) + i];
                        output_b[3 * (xp * W + yp+15) + i] += weight_15 * color[3 * (xq * W + yq+15) + i];
                        */
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < H - R - f_r; yp+=4) {
        
            scalar w_0 = weight_sum[3 * (xp * W + yp)];
            scalar w_1 = weight_sum[3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[3 * (xp * W + yp+3)];

            for (int i=0; i<3; i++){
                output_r[3 * (xp * W + yp) + i] /= w_0;
                output_r[3 * (xp * W + yp+1) + i] /= w_1;
                output_r[3 * (xp * W + yp+2) + i] /= w_2;
                output_r[3 * (xp * W + yp+3) + i] /= w_3;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < H - R - f_g; yp+=4) {
        
            scalar w_0 = weight_sum[1 + 3 * (xp * W + yp)];
            scalar w_1 = weight_sum[1 + 3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[1 + 3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[1 + 3 * (xp * W + yp+3)];
            /*
            scalar w_4 = weight_sum[1 + 3 * (xp * W + yp+4)];
            scalar w_5 = weight_sum[1 + 3 * (xp * W + yp+5)];
            scalar w_6 = weight_sum[1 + 3 * (xp * W + yp+6)];
            scalar w_7 = weight_sum[1 + 3 * (xp * W + yp+7)];
            scalar w_8 = weight_sum[1 + 3 * (xp * W + yp+8)];
            scalar w_9 = weight_sum[1 + 3 * (xp * W + yp+9)];
            scalar w_10 = weight_sum[1 + 3 * (xp * W + yp+10)];
            scalar w_11 = weight_sum[1 + 3 * (xp * W + yp+11)];
            scalar w_12 = weight_sum[1 + 3 * (xp * W + yp+12)];
            scalar w_13 = weight_sum[1 + 3 * (xp * W + yp+13)];
            scalar w_14 = weight_sum[1 + 3 * (xp * W + yp+14)];
            scalar w_15 = weight_sum[1 + 3 * (xp * W + yp+15)];
            */

            for (int i=0; i<3; i++){
                output_g[3 * (xp * W + yp) + i] /= w_0;
                output_g[3 * (xp * W + yp+1) + i] /= w_1;
                output_g[3 * (xp * W + yp+2) + i] /= w_2;
                output_g[3 * (xp * W + yp+3) + i] /= w_3;
                /*
                output_g[3 * (xp * W + yp+4) + i] /= w_4;
                output_g[3 * (xp * W + yp+5) + i] /= w_5;
                output_g[3 * (xp * W + yp+6) + i] /= w_6;
                output_g[3 * (xp * W + yp+7) + i] /= w_7;
                output_g[3 * (xp * W + yp+8) + i] /= w_8;
                output_g[3 * (xp * W + yp+9) + i] /= w_9;
                output_g[3 * (xp * W + yp+10) + i] /= w_10;
                output_g[3 * (xp * W + yp+11) + i] /= w_11;
                output_g[3 * (xp * W + yp+12) + i] /= w_12;
                output_g[3 * (xp * W + yp+13) + i] /= w_13;
                output_g[3 * (xp * W + yp+14) + i] /= w_14;
                output_g[3 * (xp * W + yp+15) + i] /= w_15;
                */
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < H - R - f_b; yp+=4) {
        
            scalar w_0 = weight_sum[2 + 3 * (xp * W + yp)];
            scalar w_1 = weight_sum[2 + 3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[2 + 3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[2 + 3 * (xp * W + yp+3)];
            /*
            scalar w_4 = weight_sum[2 + 3 * (xp * W + yp+4)];
            scalar w_5 = weight_sum[2 + 3 * (xp * W + yp+5)];
            scalar w_6 = weight_sum[2 + 3 * (xp * W + yp+6)];
            scalar w_7 = weight_sum[2 + 3 * (xp * W + yp+7)];
            scalar w_8 = weight_sum[2 + 3 * (xp * W + yp+8)];
            scalar w_9 = weight_sum[2 + 3 * (xp * W + yp+9)];
            scalar w_10 = weight_sum[2 + 3 * (xp * W + yp+10)];
            scalar w_11 = weight_sum[2 + 3 * (xp * W + yp+11)];
            scalar w_12 = weight_sum[2 + 3 * (xp * W + yp+12)];
            scalar w_13 = weight_sum[2 + 3 * (xp * W + yp+13)];
            scalar w_14 = weight_sum[2 + 3 * (xp * W + yp+14)];
            scalar w_15 = weight_sum[2 + 3 * (xp * W + yp+15)];
            */

            for (int i=0; i<3; i++){
                output_b[3 * (xp * W + yp) + i] /= w_0;
                output_b[3 * (xp * W + yp+1) + i] /= w_1;
                output_b[3 * (xp * W + yp+2) + i] /= w_2;
                output_b[3 * (xp * W + yp+3) + i] /= w_3;
                /*
                output_b[3 * (xp * W + yp+4) + i] /= w_4;
                output_b[3 * (xp * W + yp+5) + i] /= w_5;
                output_b[3 * (xp * W + yp+6) + i] /= w_6;
                output_b[3 * (xp * W + yp+7) + i] /= w_7;
                output_b[3 * (xp * W + yp+8) + i] /= w_8;
                output_b[3 * (xp * W + yp+9) + i] /= w_9;
                output_b[3 * (xp * W + yp+10) + i] /= w_10;
                output_b[3 * (xp * W + yp+11) + i] /= w_11;
                output_b[3 * (xp * W + yp+12) + i] /= w_12;
                output_b[3 * (xp * W + yp+13) + i] /= w_13;
                output_b[3 * (xp * W + yp+14) + i] /= w_14;
                output_b[3 * (xp * W + yp+15) + i] /= w_15;
                */
            }
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < R + f_r; yp++){
            for (int i = 0; i < 3; i++){
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < R + f_r; xp++){
        for (int yp = R + f_r ; yp < H - R - f_r; yp++){
            for (int i = 0; i < 3; i++){
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
            }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < R + f_g; yp++){
            for (int i = 0; i < 3; i++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < R + f_g; xp++){
        for (int yp = R + f_g ; yp < H - R - f_g; yp++){
            for (int i = 0; i < 3; i++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
            }
        }
    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2_r);
    free(temp2_g);
    free(features_weights_r);
    free(features_weights_b);
    free(gradients);

}
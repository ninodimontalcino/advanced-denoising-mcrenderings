#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void candidate_filtering_all_SSA(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, int R, int W, int H){

    // Get parameters
#define F_R 1
#define F_G 3
#define F_B 1
#define TAU_R 0.001
#define TAU_G 0.001
#define TAU_B 0.0001
#define KC_SQUARED_R 4.0
#define KF_SQUARED_R 0.36
#define KC_SQUARED_G 4.0
#define KF_SQUARED_G 0.36
#define KF_SQUARED_B 0.36

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

    for(int x =  R+F_R; x < W - R - F_R; ++x) {
        for(int y =  R+F_R; y < H -  R - F_R; ++y) {
                
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

                gradients[3 * (x * W + y) + 0] = fmin(diffL_01*diffL_01, diffR_00*diffR_00) + fmin(diffU_00*diffU_00, diffD_00*diffD_00);
                gradients[3 * (x * W + y) + 1] = fmin(diffL_01*diffL_01, diffR_01*diffR_01) + fmin(diffU_01*diffU_01, diffD_01*diffD_01);
                gradients[3 * (x * W + y) + 2] = fmin(diffL_02*diffL_02, diffR_02*diffR_02) + fmin(diffU_02*diffU_02, diffD_02*diffD_02);
        } 
    }

    // Precompute size of neighbourhood
    scalar neigh_r = 3*(2*F_R+1)*(2*F_R+1);
    scalar neigh_g = 3*(2*F_G+1)*(2*F_G+1);
    scalar neigh_b = 3*(2*F_B+1)*(2*F_B+1);

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

                    scalar normalization_r_00 = EPSILON + KC_SQUARED_R*(var_term_00);
                    scalar normalization_r_01 = EPSILON + KC_SQUARED_R*(var_term_01);
                    scalar normalization_r_02 = EPSILON + KC_SQUARED_R*(var_term_02);

                    scalar dist_var_00 = sqdist_00 - var_cancel_00;
                    scalar dist_var_01 = sqdist_01 - var_cancel_01;
                    scalar dist_var_02 = sqdist_02 - var_cancel_02;

                    temp[xp * W + yp] = ((dist_var_00) / normalization_r_00) + ((dist_var_01) / normalization_r_01) + ((dist_var_02) / normalization_r_02);
                    

                }
            }

            // Precompute feature weights
            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {
                    
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

                        scalar normalization_r_0 = KF_SQUARED_R*fmax(TAU_R, var_max_0);
                        scalar normalization_r_1 = KF_SQUARED_R*fmax(TAU_R, var_max_1);
                        scalar normalization_r_2 = KF_SQUARED_R*fmax(TAU_R, var_max_2);
                        scalar normalization_r_3 = KF_SQUARED_R*fmax(TAU_R, var_max_3);

                        scalar normalization_b_0 = KF_SQUARED_B*fmax(TAU_B, var_max_0);
                        scalar normalization_b_1 = KF_SQUARED_B*fmax(TAU_B, var_max_1);
                        scalar normalization_b_2 = KF_SQUARED_B*fmax(TAU_B, var_max_2);
                        scalar normalization_b_3 = KF_SQUARED_B*fmax(TAU_B, var_max_3);

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
                for(int yp = R + F_R; yp < H - R - F_R; yp+=8) {
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

                    for (int k=-F_R; k<=F_R; k++){
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
            for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=4) {

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

                    for (int k=-F_R; k<=F_R; k++){
                        sum_0 += temp2_r[(xp+k)*W + yp];
                        sum_1 += temp2_r[(xp+k)*W + yp+1];
                        sum_2 += temp2_r[(xp+k)*W + yp+2];
                        sum_3 += temp2_r[(xp+k)*W + yp+3];
                        /*
                        sum_4 += temp2_r[(xp+k)*W + yp+4];
                        sum_5 += temp2_r[(xp+k)*W + yp+5];
                        sum_6 += temp2_r[(xp+k)*W + yp+6];
                        sum_7 += temp2_r[(xp+k)*W + yp+7];
                        sum_8 += temp2_r[(xp+k)*W + yp+8];
                        sum_9 += temp2_r[(xp+k)*W + yp+9];
                        sum_10 += temp2_r[(xp+k)*W + yp+10];
                        sum_11 += temp2_r[(xp+k)*W + yp+11];
                        sum_12 += temp2_r[(xp+k)*W + yp+12];
                        sum_13 += temp2_r[(xp+k)*W + yp+13];
                        sum_14 += temp2_r[(xp+k)*W + yp+14];
                        sum_15 += temp2_r[(xp+k)*W + yp+15];
                        */
                    }

                    scalar color_weight_0 = exp(-fmax(0.f, (sum_0 / neigh_r)));
                    scalar color_weight_1 = exp(-fmax(0.f, (sum_1 / neigh_r)));
                    scalar color_weight_2 = exp(-fmax(0.f, (sum_2 / neigh_r)));
                    scalar color_weight_3 = exp(-fmax(0.f, (sum_3 / neigh_r)));
                    /*
                    scalar color_weight_4 = exp(-fmax(0.f, (sum_4 / neigh_r)));
                    scalar color_weight_5 = exp(-fmax(0.f, (sum_5 / neigh_r)));
                    scalar color_weight_6 = exp(-fmax(0.f, (sum_6 / neigh_r)));
                    scalar color_weight_7 = exp(-fmax(0.f, (sum_7 / neigh_r)));
                    scalar color_weight_8 = exp(-fmax(0.f, (sum_8 / neigh_r)));
                    scalar color_weight_9 = exp(-fmax(0.f, (sum_9 / neigh_r)));
                    scalar color_weight_10 = exp(-fmax(0.f, (sum_10 / neigh_r)));
                    scalar color_weight_11 = exp(-fmax(0.f, (sum_11 / neigh_r)));
                    scalar color_weight_12 = exp(-fmax(0.f, (sum_12 / neigh_r)));
                    scalar color_weight_13 = exp(-fmax(0.f, (sum_13 / neigh_r)));
                    scalar color_weight_14 = exp(-fmax(0.f, (sum_14 / neigh_r)));
                    scalar color_weight_15 = exp(-fmax(0.f, (sum_15 / neigh_r)));
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
                    
                    weight_sum[3 * (xp * W + yp)] += weight_0;
                    weight_sum[3 * (xp * W + yp+1)] += weight_1;
                    weight_sum[3 * (xp * W + yp+2)] += weight_2;
                    weight_sum[3 * (xp * W + yp+3)] += weight_3;
                    /*
                    weight_sum[3 * (xp * W + yp+4)] += weight_4;
                    weight_sum[3 * (xp * W + yp+5)] += weight_5;
                    weight_sum[3 * (xp * W + yp+6)] += weight_6;
                    weight_sum[3 * (xp * W + yp+7)] += weight_7;
                    weight_sum[3 * (xp * W + yp+8)] += weight_8;
                    weight_sum[3 * (xp * W + yp+9)] += weight_9;
                    weight_sum[3 * (xp * W + yp+10)] += weight_10;
                    weight_sum[3 * (xp * W + yp+11)] += weight_11;
                    weight_sum[3 * (xp * W + yp+12)] += weight_12;
                    weight_sum[3 * (xp * W + yp+13)] += weight_13;
                    weight_sum[3 * (xp * W + yp+14)] += weight_14;
                    weight_sum[3 * (xp * W + yp+15)] += weight_15;
                    */
                    
                    for (int i=0; i<3; i++){
                        output_r[3 * (xp * W + yp) + i] += weight_0 * color[3 * (xq * W + yq) + i];
                        output_r[3 * (xp * W + yp+1) + i] += weight_1 * color[3 * (xq * W + yq+1) + i];
                        output_r[3 * (xp * W + yp+2) + i] += weight_2 * color[3 * (xq * W + yq+2) + i];
                        output_r[3 * (xp * W + yp+3) + i] += weight_3 * color[3 * (xq * W + yq+3) + i];
                        /*
                        output_r[3 * (xp * W + yp+4) + i] += weight_4 * color[3 * (xq * W + yq+4) + i];
                        output_r[3 * (xp * W + yp+5) + i] += weight_5 * color[3 * (xq * W + yq+5) + i];
                        output_r[3 * (xp * W + yp+6) + i] += weight_6 * color[3 * (xq * W + yq+6) + i];
                        output_r[3 * (xp * W + yp+7) + i] += weight_7 * color[3 * (xq * W + yq+7) + i];
                        output_r[3 * (xp * W + yp+8) + i] += weight_8 * color[3 * (xq * W + yq+8) + i];
                        output_r[3 * (xp * W + yp+9) + i] += weight_9 * color[3 * (xq * W + yq+9) + i];
                        output_r[3 * (xp * W + yp+10) + i] += weight_10 * color[3 * (xq * W + yq+10) + i];
                        output_r[3 * (xp * W + yp+11) + i] += weight_11 * color[3 * (xq * W + yq+11) + i];
                        output_r[3 * (xp * W + yp+12) + i] += weight_12 * color[3 * (xq * W + yq+12) + i];
                        output_r[3 * (xp * W + yp+13) + i] += weight_13 * color[3 * (xq * W + yq+13) + i];
                        output_r[3 * (xp * W + yp+14) + i] += weight_14 * color[3 * (xq * W + yq+14) + i];
                        output_r[3 * (xp * W + yp+15) + i] += weight_15 * color[3 * (xq * W + yq+15) + i];
                        */
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=8) {
                    
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

                    for (int k=-F_G; k<=F_G; k++){
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
            for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {

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

                    for (int k=-F_G; k<=F_G; k++){
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

            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {

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
    for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
        for(int yp = R + F_R; yp < H - R - F_R; yp+=4) {
        
            scalar w_0 = weight_sum[3 * (xp * W + yp)];
            scalar w_1 = weight_sum[3 * (xp * W + yp+1)];
            scalar w_2 = weight_sum[3 * (xp * W + yp+2)];
            scalar w_3 = weight_sum[3 * (xp * W + yp+3)];
            /*
            scalar w_4 = weight_sum[3 * (xp * W + yp+4)];
            scalar w_5 = weight_sum[3 * (xp * W + yp+5)];
            scalar w_6 = weight_sum[3 * (xp * W + yp+6)];
            scalar w_7 = weight_sum[3 * (xp * W + yp+7)];
            scalar w_8 = weight_sum[3 * (xp * W + yp+8)];
            scalar w_9 = weight_sum[3 * (xp * W + yp+9)];
            scalar w_10 = weight_sum[3 * (xp * W + yp+10)];
            scalar w_11 = weight_sum[3 * (xp * W + yp+11)];
            scalar w_12 = weight_sum[3 * (xp * W + yp+12)];
            scalar w_13 = weight_sum[3 * (xp * W + yp+13)];
            scalar w_14 = weight_sum[3 * (xp * W + yp+14)];
            scalar w_15 = weight_sum[3 * (xp * W + yp+15)];
            */

            for (int i=0; i<3; i++){
                output_r[3 * (xp * W + yp) + i] /= w_0;
                output_r[3 * (xp * W + yp+1) + i] /= w_1;
                output_r[3 * (xp * W + yp+2) + i] /= w_2;
                output_r[3 * (xp * W + yp+3) + i] /= w_3;
                /*
                output_r[3 * (xp * W + yp+4) + i] /= w_4;
                output_r[3 * (xp * W + yp+5) + i] /= w_5;
                output_r[3 * (xp * W + yp+6) + i] /= w_6;
                output_r[3 * (xp * W + yp+7) + i] /= w_7;
                output_r[3 * (xp * W + yp+8) + i] /= w_8;
                output_r[3 * (xp * W + yp+9) + i] /= w_9;
                output_r[3 * (xp * W + yp+10) + i] /= w_10;
                output_r[3 * (xp * W + yp+11) + i] /= w_11;
                output_r[3 * (xp * W + yp+12) + i] /= w_12;
                output_r[3 * (xp * W + yp+13) + i] /= w_13;
                output_r[3 * (xp * W + yp+14) + i] /= w_14;
                output_r[3 * (xp * W + yp+15) + i] /= w_15;
                */
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
        for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {
        
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
   for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
        for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {
        
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
        for(int yp = 0; yp < R + F_R; yp++){
            for (int i = 0; i < 3; i++){
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < R + F_R; xp++){
        for (int yp = R + F_R ; yp < H - R - F_R; yp++){
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
        for(int yp = 0; yp < R + F_G; yp++){
            for (int i = 0; i < 3; i++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
    }
    for(int xp = 0; xp < R + F_G; xp++){
        for (int yp = R + F_G ; yp < H - R - F_G; yp++){
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
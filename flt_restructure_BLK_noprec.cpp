#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"
#include <immintrin.h>
#include "avx_mathfun.h"



void candidate_filtering_all_VEC_BLK_noprec(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int X0, int Y0, int B_TYPE, int B_SIZE){


    int R = p[0].r;
    int NEIGH_SIZE = (2*R+1) * (2*R+1);

    // LOAD EPSILON => USED TO AVOID DIV BY 0
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    // Get parameters
    int F_R = p[0].f;
    int F_G = p[1].f;
    int F_B = p[2].f;
    scalar tau_r = p[0].tau;
    scalar tau_g = p[1].tau;
    scalar tau_b = p[2].tau;
    scalar k_c_squared_r = p[0].kc * p[0].kc;
    scalar k_f_squared_r = p[0].kf * p[0].kf;
    scalar k_c_squared_g = p[1].kc * p[1].kc;
    scalar k_f_squared_g = p[1].kf * p[1].kf;
    scalar k_f_squared_b = p[2].kf * p[2].kf;

    const __m256 tau_r_vec = _mm256_set1_ps(tau_r);
    const __m256 tau_b_vec = _mm256_set1_ps(tau_b);
    const __m256 k_c_squared_r_vec = _mm256_set1_ps(k_c_squared_r);
    const __m256 k_f_squared_r_vec = _mm256_set1_ps(k_f_squared_r);
    const __m256 k_f_squared_b_vec = _mm256_set1_ps(k_f_squared_b);

    int f_max = fmax(F_R, fmax(F_G, F_B));
    int f_min = fmin(F_R, fmin(F_G, F_B));



    // ========================== CASE DISTINCTION =========================================================================================
    
    int COLOR_WEIGHT_START_X, COLOR_WEIGHT_END_X;
    int COLOR_WEIGHT_START_Y, COLOR_WEIGHT_END_Y;
    int FEATURE_WEIGHT_START_X, FEATURE_WEIGHT_END_X;
    int FEATURE_WEIGHT_START_Y, FEATURE_WEIGHT_END_Y;
    int R_CONV_H_START_X, R_CONV_H_END_X;
    int R_CONV_H_START_Y, R_CONV_H_END_Y;
    int R_CONV_W_START_X, R_CONV_W_END_X;
    int R_CONV_W_START_Y, R_CONV_W_END_Y;
    int G_CONV_H_START_X, G_CONV_H_END_X;
    int G_CONV_H_START_Y, G_CONV_H_END_Y;
    int G_CONV_W_START_X, G_CONV_W_END_X;
    int G_CONV_W_START_Y, G_CONV_W_END_Y;
    int B_CONV_START_X, B_CONV_END_X;
    int B_CONV_START_Y, B_CONV_END_Y;
    int B_START_X, B_END_X;
    int B_START_Y, B_END_Y;
    
    switch(B_TYPE)
    {
        case LT:
            B_START_X = X0 + R + f_max;
            B_END_X = X0 + B_SIZE;
            B_START_Y = Y0 + R + f_max;
            B_END_Y = Y0 + B_SIZE;
            
            break;

        case LL:
            B_START_X = X0 + R + f_max;
            B_END_X = X0 + B_SIZE;
            B_START_Y = Y0;
            B_END_Y = Y0 + B_SIZE;

            break;

        case LB:
            B_START_X = X0 + R + f_max;
            B_END_X = X0 + B_SIZE;
            B_START_Y = Y0;
            B_END_Y = Y0 + B_SIZE - R - f_max;

            break;
        
        case TT:
            B_START_X = X0;
            B_END_X = X0 + B_SIZE;
            B_START_Y = Y0 + R + f_max;
            B_END_Y = Y0 + B_SIZE;
            
            break;

        case II:
            B_START_X = X0;
            B_END_X = X0 + B_SIZE;
            B_START_Y = Y0;
            B_END_Y = Y0 + B_SIZE;
            
            break;

        case BB:
            B_START_X = X0;
            B_END_X = X0 + B_SIZE;
            B_START_Y = Y0;
            B_END_Y = Y0 + B_SIZE - R - f_max;

            break;

        case RT:
            B_START_X = X0;
            B_END_X = X0 + B_SIZE - R - f_max;
            B_START_Y = Y0 + R + f_max;
            B_END_Y = Y0 + B_SIZE;

            break;

        case RR:
            B_START_X = X0;
            B_END_X = X0 + B_SIZE - R - f_max;
            B_START_Y = Y0;
            B_END_Y = Y0 + B_SIZE;

            break;

        case RB:
            B_START_X = X0;
            B_END_X = X0 + B_SIZE - R - f_max;
            B_START_Y = Y0;
            B_END_Y = Y0 + B_SIZE - R - f_max;

            break;

        // operator doesn't match any case constant +, -, *, /
        default:
            printf("Error! operator is not correct");
    }

    // COLOR WEIGHT LOOP
    COLOR_WEIGHT_START_X = B_START_X - f_max;
    COLOR_WEIGHT_END_X = B_END_X + f_max;
    COLOR_WEIGHT_START_Y = B_START_Y - f_max;
    COLOR_WEIGHT_END_Y = B_END_Y + f_max;

    // FEATURE WEIGHT LOOP
    FEATURE_WEIGHT_START_X = B_START_X;
    FEATURE_WEIGHT_END_X = B_END_X;
    FEATURE_WEIGHT_START_Y = B_START_Y;
    FEATURE_WEIGHT_END_Y = B_END_Y;

    // R CONVOLUTION
    R_CONV_H_START_X = B_START_X - F_R;
    R_CONV_H_END_X = B_END_X + F_R;
    R_CONV_H_START_Y = B_START_Y;
    R_CONV_H_END_Y = B_END_Y;

    R_CONV_W_START_X = B_START_X;
    R_CONV_W_END_X = B_END_X;
    R_CONV_W_START_Y = B_START_Y;
    R_CONV_W_END_Y = B_END_Y;

    // G CONVOLUTION
    G_CONV_H_START_X = B_START_X - F_G;
    G_CONV_H_END_X = B_END_X + F_G;
    G_CONV_H_START_Y = B_START_Y;
    G_CONV_H_END_Y = B_END_Y;

    G_CONV_W_START_X = B_START_X;
    G_CONV_W_END_X = B_END_X;
    G_CONV_W_START_Y = B_START_Y;
    G_CONV_W_END_Y = B_END_Y;

    // B 
    B_CONV_START_X = B_START_X;
    B_CONV_END_X = B_END_X;
    B_CONV_START_Y = B_START_Y;
    B_CONV_END_Y = B_END_Y;


    // =====================================================================================================================================

    // (A) GRADIENT COMPUTATION
    // ---------------------------------
    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1_vec, tmp_2_vec;
    scalar features_sca, diffL_sqr_sca, diffR_sqr_sca, diffU_sqr_sca, diffD_sqr_sca, tmp_1_sca, tmp_2_sca;
    scalar *gradients;
    gradients = (scalar*) malloc(3 * B_SIZE * B_SIZE * sizeof(scalar));

    for(int i=0; i<3;++i) {
        for(int x =  B_START_X; x < B_END_X; ++x) {
            int y = B_START_Y;
            for( ; y < B_END_Y-7; y+=8) {
                
                // (1) Loading
                features_vec  = _mm256_loadu_ps(features[i][x] + y);
                diffL_sqr_vec = _mm256_loadu_ps(features[i][x-1] + y);
                diffR_sqr_vec = _mm256_loadu_ps(features[i][x+1]+ y);
                diffU_sqr_vec = _mm256_loadu_ps(features[i][x] + y-1);
                diffD_sqr_vec = _mm256_loadu_ps(features[i][x] + y+1);

                // (2) Computing Squared Differences
                diffL_sqr_vec = _mm256_sub_ps(features_vec, diffL_sqr_vec);
                diffR_sqr_vec = _mm256_sub_ps(features_vec, diffR_sqr_vec);
                diffU_sqr_vec = _mm256_sub_ps(features_vec, diffU_sqr_vec);
                diffD_sqr_vec = _mm256_sub_ps(features_vec, diffD_sqr_vec);

                diffL_sqr_vec = _mm256_mul_ps(diffL_sqr_vec, diffL_sqr_vec);
                diffR_sqr_vec = _mm256_mul_ps(diffR_sqr_vec, diffR_sqr_vec);
                diffU_sqr_vec = _mm256_mul_ps(diffU_sqr_vec, diffU_sqr_vec);
                diffD_sqr_vec = _mm256_mul_ps(diffD_sqr_vec, diffD_sqr_vec);

                // (3) Final Gradient Computation
                tmp_1_vec = _mm256_min_ps(diffL_sqr_vec, diffR_sqr_vec);
                tmp_2_vec = _mm256_min_ps(diffU_sqr_vec, diffD_sqr_vec);

                tmp_1_vec = _mm256_add_ps(tmp_1_vec, tmp_2_vec);

                _mm256_storeu_ps(gradients+i * B_SIZE*B_SIZE + (x-B_START_X) * B_SIZE + y-B_START_Y, tmp_1_vec);

            } 
            for( ; y < B_END_Y; y++) {
                
                // (1) Loading
                features_sca  = *(features[i][x] + y);
                diffL_sqr_sca = *(features[i][x-1] + y);
                diffR_sqr_sca = *(features[i][x+1]+ y);
                diffU_sqr_sca = *(features[i][x] + y-1);
                diffD_sqr_sca = *(features[i][x] + y+1);

                // (2) Computing Squared Differences
                diffL_sqr_sca = (features_sca - diffL_sqr_sca);
                diffR_sqr_sca = (features_sca - diffR_sqr_sca);
                diffU_sqr_sca = (features_sca - diffU_sqr_sca);
                diffD_sqr_sca = (features_sca - diffD_sqr_sca);

                diffL_sqr_sca = (diffL_sqr_sca * diffL_sqr_sca);
                diffR_sqr_sca = (diffR_sqr_sca * diffR_sqr_sca);
                diffU_sqr_sca = (diffU_sqr_sca * diffU_sqr_sca);
                diffD_sqr_sca = (diffD_sqr_sca * diffD_sqr_sca);

                // (3) Final Gradient Computation
                tmp_1_sca = fmin(diffL_sqr_sca, diffR_sqr_sca);
                tmp_2_sca = fmin(diffU_sqr_sca, diffD_sqr_sca);

                tmp_1_sca = (tmp_1_sca + tmp_2_sca);

                *(gradients+i * B_SIZE*B_SIZE + (x-B_START_X) * B_SIZE + y-B_START_Y) = tmp_1_sca;

            } 
        }
    }

    // (B) GLOBAL MEMORY ALLOCATION
    // ---------------------------------

    // (a) Feature Weights
    scalar* features_weights_r;
    scalar* features_weights_b;
    features_weights_r = (scalar*) malloc(B_SIZE * B_SIZE * sizeof(scalar));
    features_weights_b = (scalar*) malloc(B_SIZE * B_SIZE * sizeof(scalar));
    memset(features_weights_r, 0, B_SIZE * B_SIZE*sizeof(scalar));
    memset(features_weights_b, 0, B_SIZE * B_SIZE*sizeof(scalar));

    // (b) Temp Arrays for Convolution
    const int TEMP_SIZE = B_SIZE + f_max + R;
    scalar* temp;
    scalar* temp2_r;
    scalar* temp2_g;
    temp = (scalar*) malloc(TEMP_SIZE * TEMP_SIZE * sizeof(scalar));
    temp2_r = (scalar*) malloc(TEMP_SIZE * TEMP_SIZE * sizeof(scalar));
    temp2_g = (scalar*) malloc(TEMP_SIZE * TEMP_SIZE * sizeof(scalar));

    // -----------------------
    // MEMORY ALLOCATION  
    // ------------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum_r;
    scalar* weight_sum_g;
    scalar* weight_sum_b;

    weight_sum_r = (scalar*) calloc(B_SIZE * B_SIZE, sizeof(scalar));
    weight_sum_g = (scalar*) calloc(B_SIZE * B_SIZE, sizeof(scalar));
    weight_sum_b = (scalar*) calloc(B_SIZE * B_SIZE, sizeof(scalar));

    // Precompute size of neighbourhood
    scalar NEIGH_R_INV = 1. / (3*(2*F_R+1)*(2*F_R+1));
    scalar NEIGH_G_INV = 1. / (3*(2*F_G+1)*(2*F_G+1));

    const __m256 neigh_inv_r_vec = _mm256_set1_ps(NEIGH_R_INV);
    const __m256 neigh_inv_min_r_vec = _mm256_set1_ps(-NEIGH_R_INV);
    const __m256 neigh_inv_g_vec = _mm256_set1_ps(NEIGH_G_INV);
    const __m256 neigh_inv_min_g_vec = _mm256_set1_ps(-NEIGH_G_INV);    


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){        
           // #######################################################################################
            // WEIGHT COMPUTATION
            // #######################################################################################

            // Compute Color Weight for all pixels with fixed r
            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, normalization_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 grad_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_r_vec, normalization_b_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight_r, feat_weight_b;
            
            memset(temp, 0, TEMP_SIZE * TEMP_SIZE * sizeof(scalar));

            for (int i=0; i<3; i++){  
                for(int xp = COLOR_WEIGHT_START_X; xp < COLOR_WEIGHT_END_X; ++xp) {
                    for(int yp = COLOR_WEIGHT_START_Y; yp < COLOR_WEIGHT_END_Y; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y));
                        c_var_p_vec = _mm256_loadu_ps(color_var[i][xp]+yp);
                        c_var_q_vec = _mm256_loadu_ps(color_var[i][xq]+yq);
                        c_p_vec = _mm256_loadu_ps(color[i][xp]+yp);
                        c_q_vec = _mm256_loadu_ps(color[i][xq]+yq);

                        normalization_vec = _mm256_add_ps(c_var_p_vec, c_var_q_vec);
                        sqdist_vec = _mm256_sub_ps(c_p_vec, c_q_vec);
                        var_cancel_vec = _mm256_min_ps(c_var_p_vec, c_var_q_vec);

                        normalization_vec = _mm256_mul_ps(k_c_squared_r_vec, normalization_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_add_ps(c_var_p_vec, var_cancel_vec);

                        normalization_vec = _mm256_add_ps(EPSILON_vec, normalization_vec);

                        sqdist_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);
                        sqdist_vec = _mm256_div_ps(sqdist_vec, normalization_vec);

                        temp_vec = _mm256_add_ps(temp_vec, sqdist_vec);
                        _mm256_storeu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y), temp_vec);
                    }
                }
            }

           
            // Precompute feature weights
            memset(features_weights_r, 0, B_SIZE * B_SIZE*sizeof(scalar));
            memset(features_weights_b, 0, B_SIZE * B_SIZE*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = FEATURE_WEIGHT_START_X; xp < FEATURE_WEIGHT_END_X; ++xp) {
                    for(int yp = FEATURE_WEIGHT_START_Y; yp < FEATURE_WEIGHT_END_Y; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients+j * B_SIZE*B_SIZE + (xp-B_START_X) * B_SIZE + yp-B_START_Y);
                        feat_weight_r = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                        feat_weight_b = _mm256_loadu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));

                        sqdist_vec = _mm256_sub_ps(features_p_vec, features_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_min_ps(features_var_q_vec, features_var_p_vec);
                        var_cancel_vec = _mm256_add_ps(features_var_p_vec, var_cancel_vec);
                        dist_var_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);                        

                        var_max_vec = _mm256_max_ps(features_var_p_vec, grad_vec);
                        normalization_r_vec = _mm256_max_ps(tau_r_vec, var_max_vec);
                        normalization_b_vec = _mm256_max_ps(tau_b_vec, var_max_vec);
                        normalization_r_vec = _mm256_mul_ps(k_f_squared_r_vec, normalization_r_vec);
                        normalization_b_vec = _mm256_mul_ps(k_f_squared_b_vec, normalization_b_vec);

                        normalization_r_vec = _mm256_div_ps(dist_var_vec, normalization_r_vec);
                        normalization_b_vec = _mm256_div_ps(dist_var_vec, normalization_b_vec);
                        feat_weight_r = _mm256_min_ps(feat_weight_r, normalization_r_vec);
                        feat_weight_b = _mm256_min_ps(feat_weight_b, normalization_b_vec);

                        _mm256_storeu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y), feat_weight_r);
                        _mm256_storeu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y), feat_weight_b);
                    } 
                }
            }

            // #######################################################################################
            // BOX FILTERING => seperability of box filter kernel => two linear operations
            // #######################################################################################
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R_CONV_H_START_X; xp < R_CONV_H_END_X; ++xp) {
                int yp = R_CONV_H_START_Y;
                for(; yp < R_CONV_H_END_Y - 31; yp+=32) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-F_R; k<=F_R; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8+k-COLOR_WEIGHT_START_Y));

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                    }

                    _mm256_storeu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y), sum_0_vec);
                    _mm256_storeu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y), sum_1_vec);
                    _mm256_storeu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y), sum_2_vec);
                    _mm256_storeu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y), sum_3_vec);
                }

                for(; yp < R_CONV_H_END_Y; yp+=16) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();

                    for (int k=-F_R; k<=F_R; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8+k-COLOR_WEIGHT_START_Y));

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
        
                    }

                    _mm256_storeu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y), sum_0_vec);
                    _mm256_storeu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y), sum_1_vec);
        
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R_CONV_W_START_X; xp < R_CONV_W_END_X; ++xp) {
                int yp = R_CONV_W_START_Y;
                for(; yp < R_CONV_W_END_Y - 31; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_R=1 => 2*f+1 = 3
                    __m256 temp2_vec_0 =   _mm256_loadu_ps(temp2_r + ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_r + ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_r + ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_r + ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_r + ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r + ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r + ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r + ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    
                    // end of unrolled summation 

                    // Calculate Final Weights
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, neigh_inv_r_vec);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, neigh_inv_r_vec);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, neigh_inv_r_vec);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, neigh_inv_r_vec);

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+0*8-FEATURE_WEIGHT_START_Y));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+1*8-FEATURE_WEIGHT_START_Y));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+2*8-FEATURE_WEIGHT_START_Y));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+3*8-FEATURE_WEIGHT_START_Y));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 0 * 8 - B_START_Y));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 1 * 8 - B_START_Y));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 2 * 8 - B_START_Y));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 3 * 8 - B_START_Y));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 0 * 8 - B_START_Y), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 1 * 8 - B_START_Y), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 2 * 8 - B_START_Y), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 3 * 8 - B_START_Y), weight_sum_vec_3);
                    
                    
                    for (int i=0; i<3; i++){
                        __m256 output_vec_0 = _mm256_loadu_ps(output_r[i][xp] + (yp + 0 * 8));
                        __m256 input_vec_0 = _mm256_loadu_ps(color[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output_r[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(color[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output_r[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(color[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output_r[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(color[i][xq] + (yq + 3 * 8));

                        output_vec_0 = _mm256_fmadd_ps(weight_vec_0, input_vec_0, output_vec_0);
                        output_vec_1 = _mm256_fmadd_ps(weight_vec_1, input_vec_1, output_vec_1);
                        output_vec_2 = _mm256_fmadd_ps(weight_vec_2, input_vec_2, output_vec_2);
                        output_vec_3 = _mm256_fmadd_ps(weight_vec_3, input_vec_3, output_vec_3);

                        _mm256_storeu_ps(output_r[i][xp] + (yp + 0 * 8), output_vec_0);
                        _mm256_storeu_ps(output_r[i][xp] + (yp + 1 * 8), output_vec_1);
                        _mm256_storeu_ps(output_r[i][xp] + (yp + 2 * 8), output_vec_2);
                        _mm256_storeu_ps(output_r[i][xp] + (yp + 3 * 8), output_vec_3);

                    }
                }

                for(; yp < R_CONV_W_END_Y-7; yp+=8) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, color_weight_vec, weight_vec, weight_sum_vec;
                    for (int k=-F_R; k<=F_R; k++){
                        tmp_vec = _mm256_loadu_ps(temp2_r + (xp+k-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y));
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    color_weight_vec = _mm256_mul_ps(sum_vec, neigh_inv_r_vec);
                    

                    // Compute final weight
                    weight_vec = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                    weight_vec = _mm256_min_ps(weight_vec, color_weight_vec);
                    weight_vec = exp256_ps(weight_vec);

                    weight_sum_vec = _mm256_loadu_ps(weight_sum_r+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum_r+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y), weight_sum_vec);
                    
                    __m256 output_r_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_r_vec = _mm256_loadu_ps(output_r[i][xp] + yp);
                        color_vec = _mm256_loadu_ps(color[i][xq] + yq);

                        output_r_vec = _mm256_fmadd_ps(weight_vec, color_vec, output_r_vec);

                        _mm256_storeu_ps(output_r[i][xp] + yp, output_r_vec);
                    }
                }
                for(; yp < R_CONV_W_END_Y; yp++) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_sca = 0;
                    scalar tmp_sca, color_weight_sca, weight_sca, weight_sum_sca;
                    for (int k=-F_R; k<=F_R; k++){
                        tmp_sca = *(temp2_r + (xp+k-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y));
                        sum_sca = sum_sca + tmp_sca;
                    }
                    color_weight_sca = sum_sca * NEIGH_R_INV;
                    

                    // Compute final weight
                    weight_sca = *(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                    weight_sca = fmin(weight_sca, color_weight_sca);
                    weight_sca = exp(weight_sca);

                    weight_sum_sca = *(weight_sum_r+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
                    weight_sum_sca = weight_sum_sca + weight_sca;
                    *(weight_sum_r+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y)) = weight_sum_sca;
                    
                    scalar output_r_sca, color_sca;
                    for (int i=0; i<3; i++){
                        output_r_sca = *(output_r[i][xp] + yp);
                        color_sca = *(color[i][xq] + yq);

                        output_r_sca = weight_sca * color_sca + output_r_sca;

                        *(output_r[i][xp] + yp) = output_r_sca;
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            for(int xp = G_CONV_H_START_X; xp < G_CONV_H_END_X; ++xp) {
                int yp = G_CONV_H_START_Y;
                for(; yp < G_CONV_H_END_Y - 63; yp+=64) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();
                    __m256 sum_4_vec = _mm256_setzero_ps();
                    __m256 sum_5_vec = _mm256_setzero_ps();
                    __m256 sum_6_vec = _mm256_setzero_ps();
                    __m256 sum_7_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+4*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+5*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+6*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+7*8+k-COLOR_WEIGHT_START_Y));

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y), sum_0_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y), sum_1_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y), sum_2_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y), sum_3_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+4*8-COLOR_WEIGHT_START_Y), sum_4_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+5*8-COLOR_WEIGHT_START_Y), sum_5_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+6*8-COLOR_WEIGHT_START_Y), sum_6_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+7*8-COLOR_WEIGHT_START_Y), sum_7_vec);
                }
                
                for(; yp < G_CONV_H_END_Y - 31; yp+=32) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8+k-COLOR_WEIGHT_START_Y));
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8+k-COLOR_WEIGHT_START_Y));

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                    }

                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y), sum_0_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y), sum_1_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y), sum_2_vec);
                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y), sum_3_vec);
                }
                
                
                for(; yp < G_CONV_W_END_Y; yp+=8) {

                    __m256 sum_0_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+k-COLOR_WEIGHT_START_Y));

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                    }

                    _mm256_storeu_ps(temp2_g + (xp-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y), sum_0_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = G_CONV_W_START_X; xp < G_CONV_W_END_X; ++xp) {
                int yp = G_CONV_W_START_Y;
                for(; yp < G_CONV_W_END_Y - 31; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    __m256 temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp-3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp-3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp-3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp-3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp-2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp-2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp-2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp-2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp-1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp+1)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp+2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp+2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp+2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp+2)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3); 

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ ((xp+3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+0*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ ((xp+3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+1*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ ((xp+3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+2*8-COLOR_WEIGHT_START_Y));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ ((xp+3)-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp+3*8-COLOR_WEIGHT_START_Y));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                                        
                    // end of unrolled summation 

                    // Calculate Final Weights
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, neigh_inv_g_vec);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, neigh_inv_g_vec);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, neigh_inv_g_vec);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, neigh_inv_g_vec);

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+0*8-FEATURE_WEIGHT_START_Y));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+1*8-FEATURE_WEIGHT_START_Y));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+2*8-FEATURE_WEIGHT_START_Y));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+3*8-FEATURE_WEIGHT_START_Y));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+0*8 - B_START_Y));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+1*8 - B_START_Y));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+2*8 - B_START_Y));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+3*8 - B_START_Y));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+0*8 - B_START_Y), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+1*8 - B_START_Y), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+2*8 - B_START_Y), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp+3*8 - B_START_Y), weight_sum_vec_3);
                    
                    
                    for (int i=0; i<3; i++){
                        __m256 output_vec_0 = _mm256_loadu_ps(output_g[i][xp] + (yp + 0 * 8));
                        __m256 input_vec_0 = _mm256_loadu_ps(color[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output_g[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(color[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output_g[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(color[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output_g[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(color[i][xq] + (yq + 3 * 8));

                        output_vec_0 = _mm256_fmadd_ps(weight_vec_0, input_vec_0, output_vec_0);
                        output_vec_1 = _mm256_fmadd_ps(weight_vec_1, input_vec_1, output_vec_1);
                        output_vec_2 = _mm256_fmadd_ps(weight_vec_2, input_vec_2, output_vec_2);
                        output_vec_3 = _mm256_fmadd_ps(weight_vec_3, input_vec_3, output_vec_3);

                        _mm256_storeu_ps(output_g[i][xp] + (yp + 0 * 8), output_vec_0);
                        _mm256_storeu_ps(output_g[i][xp] + (yp + 1 * 8), output_vec_1);
                        _mm256_storeu_ps(output_g[i][xp] + (yp + 2 * 8), output_vec_2);
                        _mm256_storeu_ps(output_g[i][xp] + (yp + 3 * 8), output_vec_3);

                    }
                }

                for(; yp < G_CONV_W_END_Y-7; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, color_weight_vec, weight_vec, weight_sum_vec;
                    for (int k=-F_G; k<=F_G; k++){
                        tmp_vec = _mm256_loadu_ps(temp2_g + (xp+k-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y));
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    color_weight_vec = _mm256_mul_ps(sum_vec, neigh_inv_g_vec);

                    // Compute final weight
                    weight_vec = _mm256_loadu_ps(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                    weight_vec = _mm256_min_ps(weight_vec, color_weight_vec);
                    weight_vec = exp256_ps(weight_vec);

                    weight_sum_vec = _mm256_loadu_ps(weight_sum_g + (xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum_g + (xp-B_START_X) * B_SIZE + ( yp - B_START_Y), weight_sum_vec);
                    
                    __m256 output_g_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_g_vec = _mm256_loadu_ps(output_g[i][xp] + yp);
                        color_vec = _mm256_loadu_ps(color[i][xq] + yq);

                        output_g_vec = _mm256_fmadd_ps(weight_vec, color_vec, output_g_vec);

                        _mm256_storeu_ps(output_g[i][xp] + yp, output_g_vec);
                    }
                }

                for(; yp < G_CONV_W_END_Y; yp++) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_sca = 0;
                    scalar tmp_sca, color_weight_sca, weight_sca, weight_sum_sca;
                    for (int k=-F_G; k<=F_G; k++){
                        tmp_sca = *(temp2_g + (xp+k-COLOR_WEIGHT_START_X) * TEMP_SIZE + (yp-COLOR_WEIGHT_START_Y));
                        sum_sca = (sum_sca + tmp_sca);
                    }
                    color_weight_sca = (sum_sca * NEIGH_G_INV);

                    // Compute final weight
                    weight_sca = *(features_weights_r + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                    weight_sca = fmin(weight_sca, color_weight_sca);
                    weight_sca = exp(weight_sca);

                    weight_sum_sca = *(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
                    weight_sum_sca = (weight_sum_sca + weight_sca);
                    *(weight_sum_g+ (xp-B_START_X) * B_SIZE + ( yp - B_START_Y)) = weight_sum_sca;
                    
                    scalar output_g_sca, color_sca;
                    for (int i=0; i<3; i++){
                        output_g_sca = *(output_g[i][xp] + yp);
                        color_sca = *(color[i][xq] + yq);

                        output_g_sca = (weight_sca * color_sca + output_g_sca);

                        *(output_g[i][xp] + yp) = output_g_sca;
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = B_CONV_START_X; xp < B_CONV_END_X; ++xp) {
                int yp = B_CONV_START_Y;
                for(; yp < B_CONV_END_Y - 31; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 weight_vec_0 = _mm256_loadu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+0*8-FEATURE_WEIGHT_START_Y));
                    __m256 weight_vec_1 = _mm256_loadu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+1*8-FEATURE_WEIGHT_START_Y));
                    __m256 weight_vec_2 = _mm256_loadu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+2*8-FEATURE_WEIGHT_START_Y));
                    __m256 weight_vec_3 = _mm256_loadu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp+3*8-FEATURE_WEIGHT_START_Y));

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 0*8 - B_START_Y));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 1*8 - B_START_Y));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 2*8 - B_START_Y));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 3*8 - B_START_Y));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 0*8 - B_START_Y), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 1*8 - B_START_Y), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 2*8 - B_START_Y), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_b+ (xp-B_START_X) * B_SIZE + ( yp + 3*8 - B_START_Y), weight_sum_vec_3);
                    

                    for (int i=0; i<3; i++){
                        __m256 output_vec_0 = _mm256_loadu_ps(output_b[i][xp] + (yp + 0 * 8));
                        __m256 input_vec_0 = _mm256_loadu_ps(color[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output_b[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(color[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output_b[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(color[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output_b[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(color[i][xq] + (yq + 3 * 8));

                        output_vec_0 = _mm256_fmadd_ps(weight_vec_0, input_vec_0, output_vec_0);
                        output_vec_1 = _mm256_fmadd_ps(weight_vec_1, input_vec_1, output_vec_1);
                        output_vec_2 = _mm256_fmadd_ps(weight_vec_2, input_vec_2, output_vec_2);
                        output_vec_3 = _mm256_fmadd_ps(weight_vec_3, input_vec_3, output_vec_3);

                        _mm256_storeu_ps(output_b[i][xp] + (yp + 0 * 8), output_vec_0);
                        _mm256_storeu_ps(output_b[i][xp] + (yp + 1 * 8), output_vec_1);
                        _mm256_storeu_ps(output_b[i][xp] + (yp + 2 * 8), output_vec_2);
                        _mm256_storeu_ps(output_b[i][xp] + (yp + 3 * 8), output_vec_3);

                    }                    
                }
                for(; yp < B_CONV_END_Y-7; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    // Compute final weight
                    __m256 weight_vec = _mm256_loadu_ps(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                    weight_vec = exp256_ps(weight_vec);

                    __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_b+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum_b+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y), weight_sum_vec);
                    
                    __m256 output_b_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_b_vec = _mm256_loadu_ps(output_b[i][xp] + yp);
                        color_vec = _mm256_loadu_ps(color[i][xq] + yq);

                        output_b_vec = _mm256_fmadd_ps(weight_vec, color_vec, output_b_vec);

                        _mm256_storeu_ps(output_b[i][xp] + yp, output_b_vec);
                    }

                } 
                for(; yp < B_CONV_END_Y; yp++) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    // Compute final weight
                    scalar weight_sca = *(features_weights_b + (xp-FEATURE_WEIGHT_START_X) * B_SIZE + (yp-FEATURE_WEIGHT_START_Y));
                    weight_sca = exp(weight_sca);

                    scalar weight_sum_sca = *(weight_sum_b+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
                    weight_sum_sca = (weight_sum_sca + weight_sca);
                    *(weight_sum_b+(xp-B_START_X) * B_SIZE + ( yp - B_START_Y)) = weight_sum_sca;
                    
                    scalar output_b_sca, color_sca;
                    for (int i=0; i<3; i++){
                        output_b_sca = *(output_b[i][xp] + yp);
                        color_sca = *(color[i][xq] + yq);

                        output_b_sca = (weight_sca * color_sca + output_b_sca);

                        *(output_b[i][xp] + yp) = output_b_sca;
                    }

                }           
            }
            
        }
    }

    // Final Weight Normalization R
    for(int xp = R_CONV_W_START_X; xp < R_CONV_W_END_X; ++xp) {
        int yp;
        for(yp = R_CONV_W_START_Y; yp < R_CONV_W_END_Y - 31; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp + 0*8 - B_START_Y));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp +1*8- B_START_Y));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp +2*8- B_START_Y));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp +3*8- B_START_Y));


            for (int i=0; i<3; i++){
                __m256 output_vec_0 = _mm256_loadu_ps(output_r[i][xp] + (yp + 0 * 8));
                __m256 output_vec_1 = _mm256_loadu_ps(output_r[i][xp] + (yp + 1 * 8));
                __m256 output_vec_2 = _mm256_loadu_ps(output_r[i][xp] + (yp + 2 * 8));
                __m256 output_vec_3 = _mm256_loadu_ps(output_r[i][xp] + (yp + 3 * 8));
                
                output_vec_0 = _mm256_div_ps(output_vec_0, weight_sum_vec_0);
                output_vec_1 = _mm256_div_ps(output_vec_1, weight_sum_vec_1);
                output_vec_2 = _mm256_div_ps(output_vec_2, weight_sum_vec_2);
                output_vec_3 = _mm256_div_ps(output_vec_3, weight_sum_vec_3);

                _mm256_storeu_ps(output_r[i][xp] + (yp + 0 * 8), output_vec_0);
                _mm256_storeu_ps(output_r[i][xp] + (yp + 1 * 8), output_vec_1);
                _mm256_storeu_ps(output_r[i][xp] + (yp + 2 * 8), output_vec_2);
                _mm256_storeu_ps(output_r[i][xp] + (yp + 3 * 8), output_vec_3);
            }
        }

        for(; yp < R_CONV_W_END_Y-7; yp+=8) {     
            
            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_r[i][xp] + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_r[i][xp] + yp, output_vec);
            }

        }
        for(; yp < R_CONV_W_END_Y; yp++) {     
            
            scalar weight_sum_sca = *(weight_sum_r + (xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
            scalar output_sca;
            for (int i=0; i<3; i++){
                output_sca = *(output_r[i][xp] + yp);
                output_sca = (output_sca / weight_sum_sca);
                *(output_r[i][xp] + yp) = output_sca;
            }

        }
    }    


    // Final Weight Normalization G
    for(int xp = G_CONV_W_START_X; xp < G_CONV_W_END_X; ++xp) {
        int yp;
        for(yp = G_CONV_W_START_Y; yp < G_CONV_W_END_Y - 31; yp+=32) {    

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g + (xp-B_START_X) * B_SIZE + ( yp + 0*8 - B_START_Y));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g + (xp-B_START_X) * B_SIZE + ( yp + 1*8 - B_START_Y));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g + (xp-B_START_X) * B_SIZE + ( yp + 2*8 - B_START_Y));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g + (xp-B_START_X) * B_SIZE + ( yp + 3*8 - B_START_Y));


            for (int i=0; i<3; i++){
                __m256 output_vec_0 = _mm256_loadu_ps(output_g[i][xp] + (yp + 0 * 8));
                __m256 output_vec_1 = _mm256_loadu_ps(output_g[i][xp] + (yp + 1 * 8));
                __m256 output_vec_2 = _mm256_loadu_ps(output_g[i][xp] + (yp + 2 * 8));
                __m256 output_vec_3 = _mm256_loadu_ps(output_g[i][xp] + (yp + 3 * 8));
                
                output_vec_0 = _mm256_div_ps(output_vec_0, weight_sum_vec_0);
                output_vec_1 = _mm256_div_ps(output_vec_1, weight_sum_vec_1);
                output_vec_2 = _mm256_div_ps(output_vec_2, weight_sum_vec_2);
                output_vec_3 = _mm256_div_ps(output_vec_3, weight_sum_vec_3);

                _mm256_storeu_ps(output_g[i][xp] + (yp + 0 * 8), output_vec_0);
                _mm256_storeu_ps(output_g[i][xp] + (yp + 1 * 8), output_vec_1);
                _mm256_storeu_ps(output_g[i][xp] + (yp + 2 * 8), output_vec_2);
                _mm256_storeu_ps(output_g[i][xp] + (yp + 3 * 8), output_vec_3);
            }
        }

        for(; yp < G_CONV_W_END_Y-7; yp+=8) {     
            
            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_g +(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_g[i][xp] + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_g[i][xp] + yp, output_vec);
            }

        }

        for(; yp < G_CONV_W_END_Y; yp++) {     
            
            scalar weight_sum_vec = *(weight_sum_g +(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
            scalar output_vec;
            for (int i=0; i<3; i++){
                output_vec = *(output_g[i][xp] + yp);
                output_vec = (output_vec / weight_sum_vec);
                *(output_g[i][xp] + yp) = output_vec;
            }

        }
    }

    // Final Weight Normalization B
    for(int xp = B_CONV_START_X; xp < B_CONV_END_X; ++xp) {
        int yp = B_CONV_START_Y;
        for(; yp < B_CONV_END_Y - 31; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b + (xp-B_START_X) * B_SIZE + ( yp + 0*8 - B_START_Y));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b + (xp-B_START_X) * B_SIZE + ( yp + 1*8 - B_START_Y));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b + (xp-B_START_X) * B_SIZE + ( yp + 2*8 - B_START_Y));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b + (xp-B_START_X) * B_SIZE + ( yp + 3*8 - B_START_Y));


            for (int i=0; i<3; i++){
                __m256 output_vec_0 = _mm256_loadu_ps(output_b[i][xp] + (yp + 0 * 8));
                __m256 output_vec_1 = _mm256_loadu_ps(output_b[i][xp] + (yp + 1 * 8));
                __m256 output_vec_2 = _mm256_loadu_ps(output_b[i][xp] + (yp + 2 * 8));
                __m256 output_vec_3 = _mm256_loadu_ps(output_b[i][xp] + (yp + 3 * 8));
                
                output_vec_0 = _mm256_div_ps(output_vec_0, weight_sum_vec_0);
                output_vec_1 = _mm256_div_ps(output_vec_1, weight_sum_vec_1);
                output_vec_2 = _mm256_div_ps(output_vec_2, weight_sum_vec_2);
                output_vec_3 = _mm256_div_ps(output_vec_3, weight_sum_vec_3);

                _mm256_storeu_ps(output_b[i][xp] + (yp + 0 * 8), output_vec_0);
                _mm256_storeu_ps(output_b[i][xp] + (yp + 1 * 8), output_vec_1);
                _mm256_storeu_ps(output_b[i][xp] + (yp + 2 * 8), output_vec_2);
                _mm256_storeu_ps(output_b[i][xp] + (yp + 3 * 8), output_vec_3);
            }
        }

        for(; yp < B_CONV_END_Y-7; yp+=8) {
            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_b +(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_b[i][xp] + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_b[i][xp] + yp, output_vec);
            }

        }

        for(; yp < B_CONV_END_Y; yp++) {     
            scalar weight_sum_sca = *(weight_sum_b +(xp-B_START_X) * B_SIZE + ( yp - B_START_Y));
            scalar output_sca;
            for (int i=0; i<3; i++){
                output_sca = *(output_b[i][xp] + yp);
                output_sca = (output_sca / weight_sum_sca);
                *(output_b[i][xp] + yp) = output_sca;
            }

        }
    }



    // Free memory
    free(weight_sum_r);
    free(weight_sum_g);
    free(weight_sum_b);
    free(temp2_g);
    free(temp2_r);
    free(temp);
    free(features_weights_b);
    free(features_weights_r);
    free(gradients);

}

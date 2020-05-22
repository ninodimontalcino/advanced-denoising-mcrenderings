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



void candidate_filtering_all_VEC_BLK(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, scalar* gradients, scalar* features_weights_r, scalar* features_weights_b, scalar* temp, scalar* temp2_r, scalar* temp2_g, Flt_parameters* p, int X0, int Y0, int B_TYPE, int B_SIZE, int W, int H){


    int WH = W*H;
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
    int B_START_X, B_END_X;
    int B_START_Y, B_END_Y;
    
    switch(B_TYPE)
    {
        case LT:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = R;
            COLOR_WEIGHT_END_X = COLOR_WEIGHT_START_X + B_SIZE;
            COLOR_WEIGHT_START_Y = R;
            COLOR_WEIGHT_END_Y = COLOR_WEIGHT_START_Y + B_SIZE;
 
            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = R + f_min;
            FEATURE_WEIGHT_END_X = FEATURE_WEIGHT_START_X + B_SIZE;
            FEATURE_WEIGHT_START_Y = R + f_min;
            FEATURE_WEIGHT_END_Y = FEATURE_WEIGHT_START_Y + B_SIZE;

            // R CONVOLUTION
            R_CONV_H_START_X = R;
            R_CONV_H_END_X = R_CONV_H_START_X + B_SIZE;
            R_CONV_H_START_Y = R + F_R;
            R_CONV_H_END_Y = R_CONV_H_START_Y + B_SIZE;

            R_CONV_W_START_X = R + F_R;
            R_CONV_W_END_X = R_CONV_W_START_X + B_SIZE;
            R_CONV_W_START_Y = R + F_R;
            R_CONV_W_END_Y = R_CONV_W_START_Y + B_SIZE;

            // G CONVOLUTION
            G_CONV_H_START_X = R;
            G_CONV_H_END_X = G_CONV_H_START_X + B_SIZE;
            G_CONV_H_START_Y = R + F_G;
            G_CONV_H_END_Y = G_CONV_H_START_Y + B_SIZE;

            G_CONV_W_START_X = R + F_G;
            G_CONV_W_END_X = G_CONV_W_START_X + B_SIZE;
            G_CONV_W_START_Y = R + F_G;
            G_CONV_W_END_Y = G_CONV_W_START_Y + B_SIZE;

            // B 
            B_START_X = R + F_B;
            B_END_X = B_START_X + B_SIZE;
            B_START_Y = R + F_B;
            B_END_Y = B_START_Y + B_SIZE;
            
            break;

        case LL:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = R;
            COLOR_WEIGHT_END_X = COLOR_WEIGHT_START_X + B_SIZE;
            COLOR_WEIGHT_START_Y = Y0 + R;
            COLOR_WEIGHT_END_Y = COLOR_WEIGHT_START_Y + B_SIZE;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = R + f_min;
            FEATURE_WEIGHT_END_X = FEATURE_WEIGHT_START_X + B_SIZE;
            FEATURE_WEIGHT_START_Y = Y0 + R + f_min;
            FEATURE_WEIGHT_END_Y = FEATURE_WEIGHT_START_Y + B_SIZE;

            // R CONVOLUTION
            R_CONV_H_START_X = R;
            R_CONV_H_END_X = R_CONV_H_START_X + B_SIZE;
            R_CONV_H_START_Y = Y0 + R + F_R;
            R_CONV_H_END_Y = R_CONV_H_START_Y + B_SIZE;

            R_CONV_W_START_X = R + F_R;
            R_CONV_W_END_X = R_CONV_W_START_X + B_SIZE;
            R_CONV_W_START_Y = Y0 + R + F_R;
            R_CONV_W_END_Y = R_CONV_W_START_Y + B_SIZE;

            // G CONVOLUTION
            G_CONV_H_START_X = R;
            G_CONV_H_END_X = G_CONV_H_START_X + B_SIZE;
            G_CONV_H_START_Y = Y0 + R + F_G;
            G_CONV_H_END_Y = G_CONV_H_START_Y + B_SIZE;

            G_CONV_W_START_X = R + F_G;
            G_CONV_W_END_X = G_CONV_W_START_X + B_SIZE;
            G_CONV_W_START_Y = Y0 + R + F_G;
            G_CONV_W_END_Y = G_CONV_W_START_Y + B_SIZE;

            // B 
            B_START_X = R + F_B;
            B_END_X = B_START_X + B_SIZE;
            B_START_Y = Y0 + R + F_B;
            B_END_Y = B_START_Y + B_SIZE;

            break;

        case LB:

            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = R;
            COLOR_WEIGHT_END_X = COLOR_WEIGHT_START_X + B_SIZE;
            COLOR_WEIGHT_START_Y = H - B_SIZE + R;
            COLOR_WEIGHT_END_Y = H - R;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = R + f_min;
            FEATURE_WEIGHT_END_X = FEATURE_WEIGHT_START_X + B_SIZE;
            FEATURE_WEIGHT_START_Y = H - B_SIZE + R + f_min;
            FEATURE_WEIGHT_END_Y = H - R - f_min;

            // R CONVOLUTION
            R_CONV_H_START_X = R;
            R_CONV_H_END_X = R_CONV_H_START_X + B_SIZE;
            R_CONV_H_START_Y = H - B_SIZE + R + F_R;
            R_CONV_H_END_Y = H - R - F_R;

            R_CONV_W_START_X = R + F_R;
            R_CONV_W_END_X = R_CONV_W_START_X + B_SIZE;
            R_CONV_W_START_Y = H - B_SIZE + R + F_R;
            R_CONV_W_END_Y = H - R - F_R;

            // G CONVOLUTION
            G_CONV_H_START_X = R;
            G_CONV_H_END_X = G_CONV_H_START_X + B_SIZE;
            G_CONV_H_START_Y = H - B_SIZE + R + F_G;
            G_CONV_H_END_Y = H - R - F_G;

            G_CONV_W_START_X = R + F_G;
            G_CONV_W_END_X = G_CONV_W_START_X + B_SIZE;
            G_CONV_W_START_Y = H - B_SIZE + R + F_G;
            G_CONV_W_END_Y = H - R - F_G;

            // B 
            B_START_X = R + F_B;
            B_END_X = B_START_X + B_SIZE;
            B_START_Y = H - B_SIZE + R + F_B;
            B_END_Y = H - R - F_B;

            break;
        
        case TT:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = X0 + R;
            COLOR_WEIGHT_END_X = COLOR_WEIGHT_START_X + B_SIZE;
            COLOR_WEIGHT_START_Y = R;
            COLOR_WEIGHT_END_Y = COLOR_WEIGHT_START_Y + B_SIZE;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = X0 + R + f_min;
            FEATURE_WEIGHT_END_X = FEATURE_WEIGHT_START_X + B_SIZE;
            FEATURE_WEIGHT_START_Y = R + f_min;
            FEATURE_WEIGHT_END_Y = FEATURE_WEIGHT_START_Y + B_SIZE;

            // R CONVOLUTION
            R_CONV_H_START_X = X0 + R;
            R_CONV_H_END_X = R_CONV_H_START_X + B_SIZE;
            R_CONV_H_START_Y = R + F_R;
            R_CONV_H_END_Y = R_CONV_H_START_Y + B_SIZE;

            R_CONV_W_START_X = X0 + R + F_R;
            R_CONV_W_END_X = R_CONV_W_START_X + B_SIZE;
            R_CONV_W_START_Y = R + F_R;
            R_CONV_W_END_Y = R_CONV_W_START_Y + B_SIZE;

            // G CONVOLUTION
            G_CONV_H_START_X = X0 + R;
            G_CONV_H_END_X = G_CONV_H_START_X + B_SIZE;
            G_CONV_H_START_Y = R + F_G;
            G_CONV_H_END_Y = G_CONV_H_START_Y + B_SIZE;

            G_CONV_W_START_X = X0 + R + F_G;
            G_CONV_W_END_X = G_CONV_W_START_X + B_SIZE;
            G_CONV_W_START_Y = R + F_G;
            G_CONV_W_END_Y = G_CONV_W_START_Y + B_SIZE;

             // B 
            B_START_X = X0 + R + F_B;
            B_END_X = B_START_X + B_SIZE;
            B_START_Y = R + F_B;
            B_END_Y = B_START_Y + B_SIZE;
            
            break;

        case II:

            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = X0 + R;
            COLOR_WEIGHT_END_X = COLOR_WEIGHT_START_X + B_SIZE;
            COLOR_WEIGHT_START_Y = Y0 + R;
            COLOR_WEIGHT_END_Y = COLOR_WEIGHT_START_Y + B_SIZE;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = X0 + R + f_min;
            FEATURE_WEIGHT_END_X = FEATURE_WEIGHT_START_X + B_SIZE;
            FEATURE_WEIGHT_START_Y = Y0 + R + f_min;
            FEATURE_WEIGHT_END_Y = FEATURE_WEIGHT_START_Y + B_SIZE;

            // R CONVOLUTION
            R_CONV_H_START_X = X0 + R;
            R_CONV_H_END_X = R_CONV_H_START_X + B_SIZE;
            R_CONV_H_START_Y = Y0 + R + F_R;
            R_CONV_H_END_Y = R_CONV_H_START_Y + B_SIZE;

            R_CONV_W_START_X = X0 + R + F_R;
            R_CONV_W_END_X = R_CONV_W_START_X + B_SIZE;
            R_CONV_W_START_Y = Y0 + R + F_R;
            R_CONV_W_END_Y = R_CONV_W_START_Y + B_SIZE;

            // G CONVOLUTION
            G_CONV_H_START_X = X0 + R;
            G_CONV_H_END_X = G_CONV_H_START_X + B_SIZE;
            G_CONV_H_START_Y = Y0 + R + F_G;
            G_CONV_H_END_Y = G_CONV_H_START_Y + B_SIZE;

            G_CONV_W_START_X = X0 + R + F_G;
            G_CONV_W_END_X = G_CONV_W_START_X + B_SIZE;
            G_CONV_W_START_Y = Y0 + R + F_G;
            G_CONV_W_END_Y = G_CONV_W_START_Y + B_SIZE;

            // B 
            B_START_X = X0 + R + F_B;
            B_END_X = B_START_X + B_SIZE;
            B_START_Y = Y0 + R + F_B;
            B_END_Y = B_START_Y + B_SIZE;
            
            break;

        case BB:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = X0 + R;
            COLOR_WEIGHT_END_X = COLOR_WEIGHT_START_X + B_SIZE;
            COLOR_WEIGHT_START_Y = H - B_SIZE + R;
            COLOR_WEIGHT_END_Y = H - R;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = X0 + R + f_min;
            FEATURE_WEIGHT_END_X = FEATURE_WEIGHT_START_X + B_SIZE;
            FEATURE_WEIGHT_START_Y = H - B_SIZE + R + f_min;
            FEATURE_WEIGHT_END_Y = H - R - f_min;

            // R CONVOLUTION
            R_CONV_H_START_X = X0 + R;
            R_CONV_H_END_X = R_CONV_H_START_X + B_SIZE;
            R_CONV_H_START_Y = H - B_SIZE + R + F_R;
            R_CONV_H_END_Y = H - R - F_R;

            R_CONV_W_START_X = X0 + R + F_R;
            R_CONV_W_END_X = R_CONV_W_START_X + B_SIZE;
            R_CONV_W_START_Y = H - B_SIZE + R + F_R;
            R_CONV_W_END_Y = H - R - F_R;

            // G CONVOLUTION
            G_CONV_H_START_X = X0 + R;
            G_CONV_H_END_X = G_CONV_H_START_X + B_SIZE;
            G_CONV_H_START_Y = H - B_SIZE + R + F_G;
            G_CONV_H_END_Y = H - R - F_G;

            G_CONV_W_START_X = X0 + R + F_G;
            G_CONV_W_END_X = G_CONV_W_START_X + B_SIZE;
            G_CONV_W_START_Y = H - B_SIZE + R + F_G;
            G_CONV_W_END_Y = H - R - F_G;

            // B 
            B_START_X = X0 + R + F_B;
            B_END_X = B_START_X + B_SIZE;
            B_START_Y = H - B_SIZE + R + F_B;
            B_END_Y = H - R - F_B;

            break;

        case RT:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = W - B_SIZE + R;
            COLOR_WEIGHT_END_X = H - R;
            COLOR_WEIGHT_START_Y = R;
            COLOR_WEIGHT_END_Y = COLOR_WEIGHT_START_Y + B_SIZE;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = W - B_SIZE + R + f_min;
            FEATURE_WEIGHT_END_X = W - R - f_min;
            FEATURE_WEIGHT_START_Y = R + f_min;
            FEATURE_WEIGHT_END_Y = FEATURE_WEIGHT_START_Y + B_SIZE;

            // R CONVOLUTION
            R_CONV_H_START_X = W - B_SIZE + R;
            R_CONV_H_END_X = W - R;
            R_CONV_H_START_Y = R + F_R;
            R_CONV_H_END_Y = R_CONV_H_START_Y + B_SIZE;

            R_CONV_W_START_X = W - B_SIZE + R + F_R;
            R_CONV_W_END_X = W - R - F_R;
            R_CONV_W_START_Y = R + F_R;
            R_CONV_W_END_Y = R_CONV_W_START_Y + B_SIZE;

            // G CONVOLUTION
            G_CONV_H_START_X = W - B_SIZE + R;
            G_CONV_H_END_X = W - R;
            G_CONV_H_START_Y = R + F_G;
            G_CONV_H_END_Y = G_CONV_H_START_Y + B_SIZE;

            G_CONV_W_START_X = W - B_SIZE + R + F_G;
            G_CONV_W_END_X = W - R - F_G;
            G_CONV_W_START_Y = R + F_G;
            G_CONV_W_END_Y = G_CONV_W_START_Y + B_SIZE;

            // B 
            B_START_X = W - B_SIZE + R + F_B;
            B_END_X = W - R - F_B;
            B_START_Y = R + F_B;
            B_END_Y = B_START_Y + B_SIZE;

            break;

        case RR:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = W - B_SIZE + R;
            COLOR_WEIGHT_END_X = H - R;
            COLOR_WEIGHT_START_Y = Y0 + R;
            COLOR_WEIGHT_END_Y = COLOR_WEIGHT_START_Y + B_SIZE;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = W - B_SIZE + R + f_min;
            FEATURE_WEIGHT_END_X = W - R - f_min;
            FEATURE_WEIGHT_START_Y = Y0 + R + f_min;
            FEATURE_WEIGHT_END_Y = FEATURE_WEIGHT_START_Y + B_SIZE;

            // R CONVOLUTION
            R_CONV_H_START_X = W - B_SIZE + R;
            R_CONV_H_END_X = W - R;
            R_CONV_H_START_Y = Y0 + R + F_R;
            R_CONV_H_END_Y = R_CONV_H_START_Y + B_SIZE;

            R_CONV_W_START_X = W - B_SIZE + R + F_R;
            R_CONV_W_END_X = W - R - F_R;
            R_CONV_W_START_Y = Y0 + R + F_R;
            R_CONV_W_END_Y = R_CONV_W_START_Y + B_SIZE;

            // G CONVOLUTION
            G_CONV_H_START_X = W - B_SIZE + R;
            G_CONV_H_END_X = W - R;
            G_CONV_H_START_Y = Y0 + R + F_G;
            G_CONV_H_END_Y = G_CONV_H_START_Y + B_SIZE;

            G_CONV_W_START_X = W - B_SIZE + R + F_G;
            G_CONV_W_END_X = W - R - F_G;
            G_CONV_W_START_Y = Y0 + R + F_G;
            G_CONV_W_END_Y = G_CONV_W_START_Y + B_SIZE;

            // B 
            B_START_X = W - B_SIZE + R + F_B;
            B_END_X = W - R - F_B;
            B_START_Y = Y0 + R + F_B;
            B_END_Y = B_START_Y + B_SIZE;

            break;

        case RB:
            // COLOR WEIGHT LOOP
            COLOR_WEIGHT_START_X = W - B_SIZE + R;
            COLOR_WEIGHT_END_X = H - R;
            COLOR_WEIGHT_START_Y = H - B_SIZE + R;
            COLOR_WEIGHT_END_Y = H - R;

            // FEATURE WEIGHT LOOP
            FEATURE_WEIGHT_START_X = W - B_SIZE + R + f_min;
            FEATURE_WEIGHT_END_X = W - R - f_min;
            FEATURE_WEIGHT_START_Y = H - B_SIZE + R + f_min;
            FEATURE_WEIGHT_END_Y = H - R - f_min;

            // R CONVOLUTION
            R_CONV_H_START_X = W - B_SIZE + R;
            R_CONV_H_END_X = W - R;
            R_CONV_H_START_Y = H - B_SIZE + R + F_R;
            R_CONV_H_END_Y = H - R - F_R;

            R_CONV_W_START_X = W - B_SIZE + R + F_R;
            R_CONV_W_END_X = W - R - F_R;
            R_CONV_W_START_Y = H - B_SIZE + R + F_R;
            R_CONV_W_END_Y = H - R - F_R;

            // G CONVOLUTION
            G_CONV_H_START_X = W - B_SIZE + R;
            G_CONV_H_END_X = W - R;
            G_CONV_H_START_Y = H - B_SIZE + R + F_G;
            G_CONV_H_END_Y = H - R - F_G;

            G_CONV_W_START_X = W - B_SIZE + R + F_G;
            G_CONV_W_END_X = W - R - F_G;
            G_CONV_W_START_Y = H - B_SIZE + R + F_G;
            G_CONV_W_END_Y = H - R - F_G;

            // B 
            B_START_X = W - B_SIZE + R + F_B;
            B_END_X = W - R - F_B;
            B_START_Y = H - B_START_X + R + F_B;
            B_END_Y = H - R - F_B;

            break;

        // operator doesn't match any case constant +, -, *, /
        default:
            printf("Error! operator is not correct");
    }


    // =====================================================================================================================================

    // -----------------------
    // MEMORY ALLOCATION  
    // ------------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum_r;
    scalar* weight_sum_g;
    scalar* weight_sum_b;

    weight_sum_r = (scalar*) calloc(W * H, sizeof(scalar));
    weight_sum_g = (scalar*) calloc(W * H, sizeof(scalar));
    weight_sum_b = (scalar*) calloc(W * H, sizeof(scalar));

    // Precompute size of neighbourhood
    scalar NEIGH_R_INV = 1. / (3*(2*F_R+1)*(2*F_R+1));
    scalar NEIGH_G_INV = 1. / (3*(2*F_G+1)*(2*F_G+1));
    int final_yp_r = H - 32 + R + F_R;
    int final_yp_g = H - 32 + R + F_G;

    const __m256 neigh_inv_r_vec = _mm256_set1_ps(NEIGH_R_INV);
    const __m256 neigh_inv_min_r_vec = _mm256_set1_ps(-NEIGH_R_INV);
    const __m256 neigh_inv_g_vec = _mm256_set1_ps(NEIGH_G_INV);
    const __m256 neigh_inv_min_g_vec = _mm256_set1_ps(-NEIGH_G_INV);    


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){


            // Computing neighbour mapping
            int N_MAPPING = (r_x + 10) * (2*R+1) + (r_y + 10);
        
           // #######################################################################################
            // WEIGHT COMPUTATION
            // #######################################################################################

            // Compute Color Weight for all pixels with fixed r
            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, normalization_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 grad_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_r_vec, normalization_b_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight_r, feat_weight_b;
            

            int i=0;

            for(int xp = COLOR_WEIGHT_START_X; xp < COLOR_WEIGHT_END_X; ++xp) {
                    for(int yp = COLOR_WEIGHT_START_Y; yp < COLOR_WEIGHT_END_Y; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_setzero_ps();
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
                        _mm256_storeu_ps(temp + N_MAPPING * WH + xp * W + yp, temp_vec);
                    }
                }

            for (int i=1; i<3; i++){  
                for(int xp = COLOR_WEIGHT_START_X; xp < COLOR_WEIGHT_END_X; ++xp) {
                    for(int yp = COLOR_WEIGHT_START_Y; yp < COLOR_WEIGHT_END_Y; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp);
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
                        _mm256_storeu_ps(temp + N_MAPPING * WH + xp * W + yp, temp_vec);
                    }
                }
            }

           
            // Precompute feature weights
            int j = 0;
            for(int xp = FEATURE_WEIGHT_START_X; xp < FEATURE_WEIGHT_END_X; ++xp) {
                for(int yp = FEATURE_WEIGHT_START_Y; yp < FEATURE_WEIGHT_END_Y; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients + j * WH + xp * W + yp);
                        feat_weight_r = _mm256_setzero_ps();;
                        feat_weight_b = _mm256_setzero_ps();;

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

                        _mm256_storeu_ps(features_weights_r + N_MAPPING * WH + xp * W + yp, feat_weight_r);
                        _mm256_storeu_ps(features_weights_b + N_MAPPING * WH + xp * W + yp, feat_weight_b);
                    } 
            }

            for(int j=1; j<NB_FEATURES;++j){
                for(int xp = FEATURE_WEIGHT_START_X; xp < FEATURE_WEIGHT_END_X; ++xp) {
                    for(int yp = FEATURE_WEIGHT_START_Y; yp < FEATURE_WEIGHT_END_Y; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients + j * WH + xp * W + yp);
                        feat_weight_r = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + yp);
                        feat_weight_b = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + yp);

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

                        _mm256_storeu_ps(features_weights_r + N_MAPPING * WH + xp * W + yp, feat_weight_r);
                        _mm256_storeu_ps(features_weights_b + N_MAPPING * WH + xp * W + yp, feat_weight_b);
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

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+3*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                    }

                    _mm256_storeu_ps(temp2_r + N_MAPPING * WH + xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_r + N_MAPPING * WH + xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_r + N_MAPPING * WH + xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_r + N_MAPPING * WH + xp * W + (yp + 3 * 8), sum_3_vec);
                }

                for(; yp < R_CONV_H_END_Y; yp+=16) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();

                    for (int k=-F_R; k<=F_R; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+1*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
        
                    }

                    _mm256_storeu_ps(temp2_r + N_MAPPING * WH + xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_r + N_MAPPING * WH + xp * W + (yp + 1 * 8), sum_1_vec);
        
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
                    __m256 temp2_vec_0 =   _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp-1) * W + (yp+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp-1) * W + (yp+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp-1) * W + (yp+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp-1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp+1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp+1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp+1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp+1) * W + (yp+3*8));
                    
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

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r + xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r + xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r + xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r + xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_r + xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_r + xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_r + xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_r + xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    
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

                for(; yp < R_CONV_W_END_Y; yp+=8) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, color_weight_vec, weight_vec, weight_sum_vec;
                    for (int k=-F_R; k<=F_R; k++){
                        tmp_vec = _mm256_loadu_ps(temp2_r + N_MAPPING * WH + (xp + k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    color_weight_vec = _mm256_mul_ps(sum_vec, neigh_inv_r_vec);
                    

                    // Compute final weight
                    weight_vec = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH +xp * W + yp);
                    weight_vec = _mm256_min_ps(weight_vec, color_weight_vec);
                    weight_vec = exp256_ps(weight_vec);

                    weight_sum_vec = _mm256_loadu_ps(weight_sum_r+0 * WH + xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum_r+0 * WH + xp * W + yp, weight_sum_vec);
                    
                    __m256 output_r_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_r_vec = _mm256_loadu_ps(output_r[i][xp] + yp);
                        color_vec = _mm256_loadu_ps(color[i][xq] + yq);

                        output_r_vec = _mm256_fmadd_ps(weight_vec, color_vec, output_r_vec);

                        _mm256_storeu_ps(output_r[i][xp] + yp, output_r_vec);
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

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+3*8)+k);
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+4*8)+k);
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+5*8)+k);
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+6*8)+k);
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+7*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 3 * 8), sum_3_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 4 * 8), sum_4_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 5 * 8), sum_5_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 6 * 8), sum_6_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 7 * 8), sum_7_vec);
                }
                
                for(; yp < G_CONV_H_END_Y - 31; yp+=32) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+3*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                    }

                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 3 * 8), sum_3_vec);
                }
                
                
                for(; yp < G_CONV_W_END_Y; yp+=8) {

                    __m256 sum_0_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                    }

                    _mm256_storeu_ps(temp2_g + N_MAPPING * WH + xp * W + (yp + 0 * 8), sum_0_vec);
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
                    __m256 temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-3) * W + (yp+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-3) * W + (yp+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-3) * W + (yp+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-3) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-2) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-2) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-2) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-2) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp-1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+2) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+2) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+2) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+2) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3); 

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+3) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+3) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+3) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+ N_MAPPING * WH+(xp+3) * W + (yp+3*8));
                    
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

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    
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

                for(; yp < G_CONV_W_END_Y; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, color_weight_vec, weight_vec, weight_sum_vec;
                    for (int k=-F_G; k<=F_G; k++){
                        tmp_vec = _mm256_loadu_ps(temp2_g + N_MAPPING * WH +  (xp + k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    color_weight_vec = _mm256_mul_ps(sum_vec, neigh_inv_g_vec);

                    // Compute final weight
                    weight_vec = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + yp);
                    weight_vec = _mm256_min_ps(weight_vec, color_weight_vec);
                    weight_vec = exp256_ps(weight_vec);

                    weight_sum_vec = _mm256_loadu_ps(weight_sum_g+0 * WH + xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum_g+0 * WH + xp * W + yp, weight_sum_vec);
                    
                    __m256 output_g_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_g_vec = _mm256_loadu_ps(output_g[i][xp] + yp);
                        color_vec = _mm256_loadu_ps(color[i][xq] + yq);

                        output_g_vec = _mm256_fmadd_ps(weight_vec, color_vec, output_g_vec);

                        _mm256_storeu_ps(output_g[i][xp] + yp, output_g_vec);
                    }

                    
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = B_START_X; xp < B_END_X; ++xp) {
                int yp = B_START_Y;
                for(; yp < B_END_Y - 31; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 weight_vec_0 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 0 * 8));
                    __m256 weight_vec_1 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 1 * 8));
                    __m256 weight_vec_2 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 2 * 8));
                    __m256 weight_vec_3 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 3 * 8));

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    

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
                for(; yp < B_END_Y; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    // Compute final weight
                    __m256 weight_vec = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + yp);
                    weight_vec = exp256_ps(weight_vec);

                    __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_b+0 * WH + xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum_b+0 * WH + xp * W + yp, weight_sum_vec);
                    
                    __m256 output_b_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_b_vec = _mm256_loadu_ps(output_b[i][xp] + yp);
                        color_vec = _mm256_loadu_ps(color[i][xq] + yq);

                        output_b_vec = _mm256_fmadd_ps(weight_vec, color_vec, output_b_vec);

                        _mm256_storeu_ps(output_b[i][xp] + yp, output_b_vec);
                    }

                }            
            }
            
        }
    }

    // Final Weight Normalization R
    for(int xp = R_CONV_W_START_X; xp < R_CONV_W_END_X; ++xp) {
        int yp;
        for(yp = R_CONV_W_START_Y; yp < R_CONV_W_END_Y - 31; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 3 * 8));


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

        for(; yp < R_CONV_W_END_Y; yp+=8) {     
            
            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_r + xp * W + yp);
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_r[i][xp] + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_r[i][xp] + yp, output_vec);
            }

        }
    }    


    // Final Weight Normalization G
    for(int xp = G_CONV_W_START_X; xp < G_CONV_W_END_X; ++xp) {
        int yp;
        for(yp = G_CONV_W_START_Y; yp < G_CONV_W_END_Y - 31; yp+=32) {    

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 3 * 8));


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

        for(; yp < G_CONV_W_END_Y; yp+=8) {     
            
            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_g +0 * WH + xp * W + yp);
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_g[i][xp] + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_g[i][xp] + yp, output_vec);
            }

        }
    }

    // Final Weight Normalization B
    for(int xp = B_START_X; xp < B_END_X; ++xp) {
        int yp = B_START_Y;
        for(; yp < B_END_Y - 31; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 3 * 8));


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

        for(; yp < B_END_Y; yp+=8) {     
            
            scalar w_0 = weight_sum_b[xp * W + yp];
            scalar w_1 = weight_sum_b[xp * W + yp+1];
            scalar w_2 = weight_sum_b[xp * W + yp+2];
            scalar w_3 = weight_sum_b[xp * W + yp+3];

            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum_b +0 * WH + xp * W + yp);
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_b[i][xp] + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_b[i][xp] + yp, output_vec);
            }

        }
    }



    // Free memory
    free(weight_sum_r);
    free(weight_sum_g);
    free(weight_sum_b);

}

// ==========================================================================================================================================

void candidate_filtering_all_VEC_BLK_PREP(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, scalar* gradients, scalar* features_weights_r, scalar* features_weights_b, scalar* temp, scalar* temp2_r, scalar* temp2_g, Flt_parameters* p, int W, int H){

    int WH = W*H;
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

    // -----------------------
    // MEMORY ALLOCATION  
    // ------------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum_r;
    scalar* weight_sum_g;
    scalar* weight_sum_b;

    weight_sum_r = (scalar*) calloc(W * H, sizeof(scalar));
    weight_sum_g = (scalar*) calloc(W * H, sizeof(scalar));
    weight_sum_b = (scalar*) calloc(W * H, sizeof(scalar));

    // Precompute size of neighbourhood
    scalar NEIGH_R_INV = 1. / (3*(2*F_R+1)*(2*F_R+1));
    scalar NEIGH_G_INV = 1. / (3*(2*F_G+1)*(2*F_G+1));
    int final_yp_r = H - 32 + R + F_R;
    int final_yp_g = H - 32 + R + F_G;

    const __m256 neigh_inv_r_vec = _mm256_set1_ps(NEIGH_R_INV);
    const __m256 neigh_inv_min_r_vec = _mm256_set1_ps(-NEIGH_R_INV);
    const __m256 neigh_inv_g_vec = _mm256_set1_ps(NEIGH_G_INV);
    const __m256 neigh_inv_min_g_vec = _mm256_set1_ps(-NEIGH_G_INV);    


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){


            // Computing neighbour mapping
            int N_MAPPING = (r_x + 10) * (2*R+1) + (r_y + 10);
        
            // #######################################################################################
            // WEIGHT COMPUTATION
            // #######################################################################################

            // Compute Color Weight for all pixels with fixed r
            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, normalization_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 grad_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_r_vec, normalization_b_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight_r, feat_weight_b;
            
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp);
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
                        _mm256_storeu_ps(temp + N_MAPPING * WH +  xp * W + yp, temp_vec);
                    }
                }
            }

           
            // Precompute feature weights
            // @Comment from Nino: Old loop order is faster, but this one is easier for vectorization => Still room for improvements

            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + f_min; xp < W - R - f_min; ++xp) {
                    for(int yp = R + f_min; yp < H - R - f_min; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients + j * WH + xp * W + yp);
                        feat_weight_r = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + yp);
                        feat_weight_b = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + yp);

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

                        _mm256_storeu_ps(features_weights_r + N_MAPPING * WH + xp * W + yp, feat_weight_r);
                        _mm256_storeu_ps(features_weights_b + N_MAPPING * WH + xp * W + yp, feat_weight_b);
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
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=64) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();
                    __m256 sum_4_vec = _mm256_setzero_ps();
                    __m256 sum_5_vec = _mm256_setzero_ps();
                    __m256 sum_6_vec = _mm256_setzero_ps();
                    __m256 sum_7_vec = _mm256_setzero_ps();

                    for (int k=-F_R; k<=F_R; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+3*8)+k);
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+4*8)+k);
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+5*8)+k);
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+6*8)+k);
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+7*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 3 * 8), sum_3_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 4 * 8), sum_4_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 5 * 8), sum_5_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 6 * 8), sum_6_vec);
                    _mm256_storeu_ps(temp2_r+ N_MAPPING * WH + xp * W + (yp + 7 * 8), sum_7_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
                for(int yp = R + F_R; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_R=1 => 2*f+1 = 3
                    __m256 temp2_vec_0 =   _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp-1) * W + (yp+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp-1) * W + (yp+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp-1) * W + (yp+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp-1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp+1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp+1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp+1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r+ N_MAPPING * WH +(xp+1) * W + (yp+3*8));
                    
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

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_r+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_r+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_r+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_r+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    
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

                for(int yp = final_yp_r; yp < H - R - F_R; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    // Unrolled Summation => Fixed for F_R=1 => 2*f+1 = 3
                    sum_0 += temp2_r[N_MAPPING * WH + (xp-1) * W + yp+0];
                    sum_1 += temp2_r[N_MAPPING * WH + (xp-1) * W + yp+1];    
                    sum_2 += temp2_r[N_MAPPING * WH + (xp-1) * W + yp+2];    
                    sum_3 += temp2_r[N_MAPPING * WH + (xp-1) * W + yp+3];

                    sum_0 += temp2_r[N_MAPPING * WH + (xp) * W + yp+0];
                    sum_1 += temp2_r[N_MAPPING * WH + (xp) * W + yp+1];
                    sum_2 += temp2_r[N_MAPPING * WH + (xp) * W + yp+2];
                    sum_3 += temp2_r[N_MAPPING * WH + (xp) * W + yp+3];

                    sum_0 += temp2_r[N_MAPPING * WH + (xp+1) * W + yp+0];
                    sum_1 += temp2_r[N_MAPPING * WH + (xp+1) * W + yp+1];
                    sum_2 += temp2_r[N_MAPPING * WH + (xp+1) * W + yp+2];
                    sum_3 += temp2_r[N_MAPPING * WH + (xp+1) * W + yp+3];


                    // Compute final weight
                    scalar weight_0 = exp(fmin((sum_0 * NEIGH_R_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 0]));
                    scalar weight_1 = exp(fmin((sum_1 * NEIGH_R_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 1]));
                    scalar weight_2 = exp(fmin((sum_2 * NEIGH_R_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 2]));
                    scalar weight_3 = exp(fmin((sum_3 * NEIGH_R_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 3]));

                    weight_sum_r[xp * W + yp + 0] += weight_0;
                    weight_sum_r[xp * W + yp + 1] += weight_1;
                    weight_sum_r[xp * W + yp + 2] += weight_2;
                    weight_sum_r[xp * W + yp + 3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp+0] += weight_0 * color[i][xq][yq+0];
                        output_r[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_r[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_r[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                    }
                    
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=64) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();
                    __m256 sum_4_vec = _mm256_setzero_ps();
                    __m256 sum_5_vec = _mm256_setzero_ps();
                    __m256 sum_6_vec = _mm256_setzero_ps();
                    __m256 sum_7_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+3*8)+k);
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+4*8)+k);
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+5*8)+k);
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+6*8)+k);
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp + N_MAPPING * WH + xp * W + (yp+7*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH +  xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 3 * 8), sum_3_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 4 * 8), sum_4_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 5 * 8), sum_5_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 6 * 8), sum_6_vec);
                    _mm256_storeu_ps(temp2_g+ N_MAPPING * WH + xp * W + (yp + 7 * 8), sum_7_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
                for(int yp = R + F_G; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    __m256 temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-3) * W + (yp+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-3) * W + (yp+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-3) * W + (yp+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-3) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-2) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-2) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-2) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-2) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp-1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+2) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+2) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+2) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+2) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3); 

                    temp2_vec_0 =   _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+3) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+3) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+3) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+N_MAPPING * WH + (xp+3) * W + (yp+3*8));
                    
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

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r + N_MAPPING * WH + xp * W + (yp + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_g+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    
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

                for(int yp = final_yp_g; yp < H - R - F_R; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    sum_0 += temp2_g[N_MAPPING * WH + (xp-3) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp-3) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp-3) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp-3) * W + yp+3];

                    sum_0 += temp2_g[N_MAPPING * WH + (xp-2) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp-2) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp-2) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp-2) * W + yp+3];

                    sum_0 += temp2_g[N_MAPPING * WH + (xp-1) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp-1) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp-1) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp-1) * W + yp+3];

                    sum_0 += temp2_g[N_MAPPING * WH + (xp) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp) * W + yp+3];

                    sum_0 += temp2_g[N_MAPPING * WH + (xp+1) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp+1) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp+1) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp+1) * W + yp+3];
                    
                    sum_0 += temp2_g[N_MAPPING * WH + (xp+2) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp+2) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp+2) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp+2) * W + yp+3];

                    sum_0 += temp2_g[N_MAPPING * WH + (xp+3) * W + yp];
                    sum_1 += temp2_g[N_MAPPING * WH + (xp+3) * W + yp+1];
                    sum_2 += temp2_g[N_MAPPING * WH + (xp+3) * W + yp+2];
                    sum_3 += temp2_g[N_MAPPING * WH + (xp+3) * W + yp+3];


                    // Compute final weight
                    scalar weight_0 = exp(fmin((sum_0 * NEIGH_G_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 0]));
                    scalar weight_1 = exp(fmin((sum_1 * NEIGH_G_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 1]));
                    scalar weight_2 = exp(fmin((sum_2 * NEIGH_G_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 2]));
                    scalar weight_3 = exp(fmin((sum_3 * NEIGH_G_INV), features_weights_r[N_MAPPING * WH + xp * W + yp + 3]));

                    weight_sum_g[xp * W + yp + 0] += weight_0;
                    weight_sum_g[xp * W + yp + 1] += weight_1;
                    weight_sum_g[xp * W + yp + 2] += weight_2;
                    weight_sum_g[xp * W + yp + 3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp+0] += weight_0 * color[i][xq][yq+0];
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

            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 weight_vec_0 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 0 * 8));
                    __m256 weight_vec_1 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 1 * 8));
                    __m256 weight_vec_2 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 2 * 8));
                    __m256 weight_vec_3 = _mm256_loadu_ps(features_weights_b + N_MAPPING * WH + xp * W + (yp + 3 * 8));

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_b+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    

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
                for(int yp = final_yp_r; yp < H - R - F_B; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar weight_0 = exp(features_weights_b[N_MAPPING * WH + xp * W + yp]);
                    scalar weight_1 = exp(features_weights_b[N_MAPPING * WH + xp * W + yp+1]);
                    scalar weight_2 = exp(features_weights_b[N_MAPPING * WH + xp * W + yp+2]);
                    scalar weight_3 = exp(features_weights_b[N_MAPPING * WH + xp * W + yp+3]);

                    weight_sum_b[xp * W + yp] += weight_0;
                    weight_sum_b[xp * W + yp+1] += weight_1;
                    weight_sum_b[xp * W + yp+2] += weight_2;
                    weight_sum_b[xp * W + yp+3] += weight_3;

                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_b[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_b[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_b[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                    }
                }            
            }
            
        }
    }

    // Final Weight Normalization R
    for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
        for(int yp = R + F_R ; yp < H - 32; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r + xp * W + (yp + 3 * 8));


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

        for(int yp = final_yp_r; yp < H - R - F_R; yp+=4) {     
            
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
    for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
        for(int yp = R + F_G ; yp < H - 32; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g + xp * W + (yp + 3 * 8));


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

        for(int yp = final_yp_g; yp < H - R - F_G; yp+=4) {     
            
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
    for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
        for(int yp = R + F_B ; yp < H - 32; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b + xp * W + (yp + 3 * 8));


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

        for(int yp = final_yp_r; yp < H - R - F_B; yp+=4) {     
            
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



    // Free memory
    free(weight_sum_r);
    free(weight_sum_g);
    free(weight_sum_b);
    free(temp2_r);
    free(temp2_g);
    free(gradients);

}

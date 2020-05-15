#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>

#include "avx_mathfun.h"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all_vec(scalar* sure, scalar* c, scalar* c_var, scalar* cand_r, scalar* cand_g, scalar* cand_b, int W, int H){

    int WH = W*H;
    
    __m256 d_r_vec, d_g_vec, d_b_vec, v_vec, c_vec;
    
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y+=8){


            __m256 sure_r_vec = _mm256_setzero_ps();
            __m256 sure_g_vec = _mm256_setzero_ps();
            __m256 sure_b_vec = _mm256_setzero_ps();

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Loading
                c_vec = _mm256_load_ps(c+(i * WH + x * W + y));
                d_r_vec = _mm256_load_ps(cand_r+(i * WH + x * W + y));
                d_g_vec = _mm256_load_ps(cand_g+(i * WH + x * W + y));
                d_b_vec = _mm256_load_ps(cand_b+(i * WH + x * W + y));
                v_vec = _mm256_load_ps(c_var+(i * WH + x * W + y));

                // d_r = d_r - c
                d_r_vec = _mm256_sub_ps(d_r_vec, c_vec);
                d_g_vec = _mm256_sub_ps(d_g_vec, c_vec);
                d_b_vec = _mm256_sub_ps(d_b_vec, c_vec);

                // Squared
                v_vec = _mm256_mul_ps(v_vec, v_vec);
                d_r_vec = _mm256_mul_ps(d_r_vec, d_r_vec);
                d_g_vec = _mm256_mul_ps(d_g_vec, d_g_vec);
                d_b_vec = _mm256_mul_ps(d_b_vec, d_b_vec);

                // Difference d_r = d_r-v
                d_r_vec = _mm256_sub_ps(d_r_vec, v_vec);
                d_g_vec = _mm256_sub_ps(d_g_vec, v_vec);
                d_b_vec = _mm256_sub_ps(d_b_vec, v_vec);

                // Summing up
                sure_r_vec = _mm256_add_ps(sure_r_vec, d_r_vec);
                sure_g_vec = _mm256_add_ps(sure_g_vec, d_g_vec);
                sure_b_vec = _mm256_add_ps(sure_b_vec, d_b_vec);

            }
            // Store sure error estimate
            _mm256_store_ps(sure+(0 * WH + x * W + y), sure_r_vec);
            _mm256_store_ps(sure+(1 * WH + x * W + y), sure_g_vec);
            _mm256_store_ps(sure+(2 * WH + x * W + y), sure_b_vec);
        }
    }
}


void filtering_basic_vec(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int W, int H){
    
    int WH = W*H;
    int ALIGN = 8;
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);


    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(k_c_squared);

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    allocate_channel_aligned(&weight_sum, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    allocate_channel_aligned(&temp, W, H);
    allocate_channel_aligned(&temp2, W, H);


    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);
    const __m256 neigh_vec = _mm256_set1_ps(neigh);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = ALIGN; xp < W - ALIGN; ++xp) {
                for(int yp = ALIGN; yp < H - ALIGN; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    __m256 distance_vec = _mm256_setzero_ps();
                    __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, var_cancel_vec, normalization_vec;
                    for (int i=0; i<3; i++){
                        c_var_p_vec = _mm256_load_ps(c_var+i * WH + xp * W + yp);
                        c_var_q_vec = _mm256_loadu_ps(c_var+i * WH + xq * W + yq);
                        c_p_vec = _mm256_load_ps(c+i * WH + xp * W + yp);
                        c_q_vec = _mm256_loadu_ps(c+ i * WH + xq * W + yq);

                        normalization_vec = _mm256_add_ps(c_var_p_vec, c_var_q_vec);
                        sqdist_vec = _mm256_sub_ps(c_p_vec, c_q_vec);
                        var_cancel_vec = _mm256_min_ps(c_var_p_vec, c_var_q_vec);

                        normalization_vec = _mm256_mul_ps(k_c_squared_vec, normalization_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_add_ps(c_var_p_vec, var_cancel_vec);

                        normalization_vec = _mm256_add_ps(EPSILON_vec, normalization_vec);

                        sqdist_vec = _mm256_sub_ps(sqdist_vec, var_cancel_vec);
                        sqdist_vec = _mm256_div_ps(sqdist_vec, normalization_vec);
                        
                        distance_vec = _mm256_add_ps(distance_vec, sqdist_vec);
                    }
                    _mm256_store_ps(temp+xp*W+yp, distance_vec);

                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = ALIGN; xp < W - ALIGN; ++xp) {
                for(int yp = ALIGN; yp < H - ALIGN; yp+=8) {

                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec;
                    for (int k=-p.f; k<=p.f; k++){
                        tmp_vec = _mm256_load_ps(temp+xp * W + yp + k);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    _mm256_store_ps(temp2+xp * W + yp, sum_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = ALIGN; xp < W - ALIGN; ++xp) {
                for(int yp = ALIGN; yp < H - ALIGN; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, weight_vec;
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        tmp_vec = _mm256_load_ps(temp2+(xp+k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    weight_vec = _mm256_div_ps(sum_vec, neigh_vec);
                    weight_vec = _mm256_max_ps(weight_vec, _mm256_setzero_ps());
                    weight_vec = _mm256_mul_ps(weight_vec, _mm256_set1_ps(-1.0));
                    weight_vec = exp256_ps(weight_vec);

                    __m256 weight_sum_vec = _mm256_load_ps(weight_sum+xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_store_ps(weight_sum+xp * W + yp, weight_sum_vec);
                    
                    __m256 input_vec, output_vec;
                    for (int i=0; i<3; i++){
                        input_vec = _mm256_loadu_ps(input+i * WH + xq * W + yq);
                        output_vec = _mm256_load_ps(output+i * WH + xp * W + yp);
                        input_vec = _mm256_mul_ps(weight_vec, input_vec);
                        output_vec = _mm256_add_ps(input_vec, output_vec);
                        _mm256_store_ps(output+i * WH + xp * W + yp, output_vec);                        
                    }
                }
            }

        }
    }

    // Final Weight Normalization
    for(int xp = ALIGN; xp < W - ALIGN; ++xp) {
        for(int yp = ALIGN ; yp < H - ALIGN; yp+=8) {     
            __m256 weight_sum_vec, output_vec;
            weight_sum_vec = _mm256_load_ps(weight_sum+xp * W + yp);
            for (int i=0; i<3; i++){
                output_vec = _mm256_load_ps(output+i * WH + xp * W + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_store_ps(output+i * WH + xp * W + yp, output_vec);
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){

            __m256 input_vec1, input_vec2;
            input_vec1 = _mm256_load_ps(input +i * WH + xp * W );
            input_vec2 = _mm256_load_ps(input +i * WH + xp * W + H - ALIGN);
            _mm256_store_ps(output+i * WH + xp * W , input_vec1);
            _mm256_store_ps(output+i * WH + xp * W + H - ALIGN, input_vec2);
                
            
        }
        for(int xp = 0; xp < ALIGN; xp++){
             for (int yp = ALIGN ; yp < H - ALIGN; yp+=8){
                // output[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                // output[i * WH + (W - xp - 1) * W + yp] = input[i * WH + (W - xp - 1) * W + yp];
                __m256 input_vec1, input_vec2;
                input_vec1 = _mm256_load_ps(input +i * WH + xp * W + yp);
                input_vec2 = _mm256_load_ps(input +i * WH + (W - xp - 1) * W + yp);
                _mm256_store_ps(output+i * WH + xp * W + yp, input_vec1);
                _mm256_store_ps(output+i * WH + (W - xp - 1) * W + yp, input_vec2);

            }
        }

    }

    free(weight_sum);
    free(temp);
    free(temp2); 
}


void feature_prefiltering_vec(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int W, int H){

    int ALIGN = 8;
    int WH = W*H;
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);
    
    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(k_c_squared);

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    allocate_channel_aligned_zero(&weight_sum, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    allocate_channel_aligned(&temp, W, H);
    allocate_channel_aligned(&temp2, W, H);

    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);
    const __m256 neigh_vec = _mm256_set1_ps(neigh);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = ALIGN; xp < W - ALIGN; ++xp) {
                for(int yp = ALIGN; yp < H - ALIGN; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    __m256 distance_vec = _mm256_setzero_ps();
                    __m256 sqdist_vec, feature_p_vec, feature_q_vec, feature_var_p_vec, feature_var_q_vec, var_cancel_vec, normalization_vec, dist_tmp;
                    for (int i=0; i<3; i++){               
                        feature_p_vec = _mm256_load_ps(features+i * WH + xp * W + yp);
                        feature_q_vec = _mm256_loadu_ps(features+i * WH + xq * W + yq);
                        sqdist_vec = _mm256_sub_ps(feature_p_vec, feature_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);         

                        feature_var_p_vec = _mm256_load_ps(features_var+i * WH + xp * W + yp);
                        feature_var_q_vec = _mm256_loadu_ps(features_var+i * WH + xq * W + yq);
                        var_cancel_vec = _mm256_min_ps(feature_var_p_vec, feature_var_q_vec);
                        var_cancel_vec = _mm256_add_ps(var_cancel_vec, feature_var_p_vec);
                        
                        normalization_vec = _mm256_add_ps(feature_var_p_vec, feature_var_q_vec);
                        normalization_vec = _mm256_mul_ps(normalization_vec, k_c_squared_vec);
                        normalization_vec = _mm256_add_ps(normalization_vec, EPSILON_vec);

                        dist_tmp = _mm256_sub_ps(sqdist_vec, var_cancel_vec);
                        dist_tmp = _mm256_div_ps(dist_tmp, normalization_vec);
                        distance_vec = _mm256_add_ps(distance_vec, dist_tmp);
                    }

                    _mm256_store_ps(temp+xp*W+yp, distance_vec);

                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=8) {
                    
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec;
                    for (int k=-p.f; k<=p.f; k++){
                        tmp_vec = _mm256_load_ps(temp+xp * W + yp + k);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    _mm256_store_ps(temp2+xp * W + yp, sum_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, weight_vec;
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        tmp_vec = _mm256_load_ps(temp2+(xp+k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    weight_vec = _mm256_div_ps(sum_vec, neigh_vec);
                    weight_vec = _mm256_max_ps(weight_vec, _mm256_setzero_ps());
                    weight_vec = _mm256_mul_ps(weight_vec, _mm256_set1_ps(-1.0));
                    weight_vec = exp256_ps(weight_vec);

                    __m256 weight_sum_vec = _mm256_load_ps(weight_sum+xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_store_ps(weight_sum+xp * W + yp, weight_sum_vec);
                    
                    __m256 features_vec, output_vec, features_var_vec, output_var_vec;
                    for (int i=0; i<3; i++){
                        features_vec = _mm256_loadu_ps(features+i * WH + xq * W + yq);
                        output_vec = _mm256_load_ps(output+i * WH + xp * W + yp);
                        features_vec = _mm256_mul_ps(weight_vec, features_vec);
                        output_vec = _mm256_add_ps(features_vec, output_vec);
                        _mm256_store_ps(output+i * WH + xp * W + yp, output_vec);
                        
                        features_var_vec = _mm256_loadu_ps(features_var+i * WH + xq * W + yq);
                        output_var_vec = _mm256_load_ps(output_var+i * WH + xp * W + yp);
                        features_var_vec = _mm256_mul_ps(weight_vec, features_var_vec);
                        output_var_vec = _mm256_add_ps(features_var_vec, output_var_vec);
                        _mm256_store_ps(output_var+i * WH + xp * W + yp, output_var_vec);                          
                    }
                    
                }
            }


        }
    }

    // Final Weight Normalization
    for(int xp = ALIGN; xp < W - ALIGN; ++xp) {
        for(int yp = ALIGN; yp < H - ALIGN; ++yp) {
        
            scalar w = weight_sum[xp * W + yp];
            for (int i=0; i<3; i++){
                output[i * WH + xp * W + yp] /= w;
                output_var[i * WH + xp * W + yp] /= w;
            }
        }
    }

    // // Final Weight Normalization
    // for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
    //     for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=8) {
        
    //         __m256 weight_sum_vec, output_vec;
    //         weight_sum_vec = _mm256_load_ps(weight_sum+xp * W + yp);
    //         for (int i=0; i<3; i++){
    //             output_vec = _mm256_load_ps(output+i * WH + xp * W + yp);
    //             output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
    //             _mm256_store_ps(output+i * WH + xp * W + yp, output_vec);
    //         }
    //     }
    // }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < ALIGN; yp++){
                output[i * WH + xp * W + yp] = features[i * WH + xp * W + yp];
                output[i * WH + xp * W + H - yp - 1] = features[i * WH + xp * W + H - yp - 1];
                output_var[i * WH + xp * W + yp] = features_var[i * WH + xp * W + yp];
                output_var[i * WH + xp * W + H - yp - 1] = features_var[i * WH + xp * W + H - yp - 1];
            }
        }
        for(int xp = 0; xp < ALIGN; xp++){
             for (int yp = ALIGN ; yp < H - ALIGN; yp++){
                output[i * WH + xp * W + yp] = features[i * WH + xp * W + yp];
                output[i * WH + (W - xp - 1) * W + yp] = features[i * WH + (W - xp - 1) * W + yp];
                output_var[i * WH + xp * W + yp] = features_var[i * WH + xp * W + yp];
                output_var[i * WH + (W - xp - 1) * W + yp] = features_var[i * WH + (W - xp - 1) * W + yp];
            }
        }

    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2); 

}



void candidate_filtering_all_vec(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int W, int H){

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

    const __m256 tau_r_vec = _mm256_set1_ps(tau_r);
    const __m256 tau_b_vec = _mm256_set1_ps(tau_b);
    const __m256 k_c_squared_r_vec = _mm256_set1_ps(k_c_squared_r);
    const __m256 k_f_squared_r_vec = _mm256_set1_ps(k_f_squared_r);
    const __m256 k_f_squared_b_vec = _mm256_set1_ps(k_f_squared_b);

    // Rename Width and Height
    int WH = W * H;

    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);


    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(f_r, fmax(f_g, f_b));
    int f_min = fmin(f_r, fmin(f_g, f_b));
    int R = p[0].r;

    // Determine alignment constants
    int ALRfmin, ALR, ALRr, ALRg, ALRb;

    if ((R + f_min) % 8) ALRfmin = R + f_min + 8 - (R + f_min) % 8;
    else ALRfmin = R + f_min;
    if (R % 8) ALR = R + 8 - R % 8;
    else ALR = R;
    if ((R + f_r) % 8) ALRr = R + f_r + 8 - (R + f_r) % 8;
    else ALRr = R + f_r; 
    if ((R + f_g) % 8) ALRg = R + f_g + 8 - (R + f_g) % 8;
    else ALRg = R + f_g;
    if ((R + f_b) % 8) ALRb = R + f_b + 8 - (R + f_b) % 8;
    else ALRb = R + f_b;


    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    scalar *weight_sum;
    allocate_buffer_aligned(&weight_sum, W, H);

    // Init temp channel
    scalar *temp;
    scalar *temp2_r;
    scalar *temp2_g;
    allocate_channel_aligned(&temp, W, H);
    allocate_channel_aligned(&temp2_r, W, H);
    allocate_channel_aligned(&temp2_g, W, H);

    // Allocate feature weights buffer
    scalar *features_weights_r;
    scalar *features_weights_b;
    allocate_channel_aligned(&features_weights_r, W, H);
    allocate_channel_aligned(&features_weights_b, W, H);

    // Compute gradients
    scalar *gradients;
    allocate_buffer_aligned(&gradients, W, H);
    //gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1, tmp_2, grad_vec;
    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  ALRfmin; x < W - ALRfmin; ++x) {
            for(int y =  ALRfmin; y < H- ALRfmin; y+=8) {

                // Loading
                features_vec  = _mm256_load_ps(features+i * WH + x * W + y);
                diffL_sqr_vec = _mm256_load_ps(features+i * WH + (x-1) * W + y);
                diffR_sqr_vec =_mm256_load_ps(features+i * WH + (x+1) * W + y);
                diffU_sqr_vec =_mm256_load_ps(features+i * WH + x * W + y-1);
                diffD_sqr_vec =_mm256_load_ps(features+i * WH + x * W + y+1);

                // Computing squared differences

                diffL_sqr_vec = _mm256_sub_ps(features_vec, diffL_sqr_vec);
                diffR_sqr_vec = _mm256_sub_ps(features_vec, diffR_sqr_vec);
                diffU_sqr_vec = _mm256_sub_ps(features_vec, diffU_sqr_vec);
                diffD_sqr_vec = _mm256_sub_ps(features_vec, diffD_sqr_vec);

                diffL_sqr_vec = _mm256_mul_ps(diffL_sqr_vec, diffL_sqr_vec);
                diffR_sqr_vec = _mm256_mul_ps(diffR_sqr_vec, diffR_sqr_vec);
                diffU_sqr_vec = _mm256_mul_ps(diffU_sqr_vec, diffU_sqr_vec);
                diffD_sqr_vec = _mm256_mul_ps(diffD_sqr_vec, diffD_sqr_vec);

                tmp_1 = _mm256_min_ps(diffL_sqr_vec, diffR_sqr_vec);
                tmp_2 = _mm256_min_ps(diffU_sqr_vec, diffD_sqr_vec);

                tmp_1 = _mm256_add_ps(tmp_1, tmp_2);

                _mm256_store_ps(gradients+i * WH + x * W + y, tmp_1);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar neigh_r = 3*(2*f_r+1)*(2*f_r+1);
    scalar neigh_g = 3*(2*f_g+1)*(2*f_g+1);
    scalar neigh_b = 3*(2*f_b+1)*(2*f_b+1);

    const __m256 neigh_r_vec = _mm256_set1_ps(neigh_r);
    const __m256 neigh_g_vec = _mm256_set1_ps(neigh_g);

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = ALR; xp < W - ALR; ++xp) {
                for(int yp = ALR; yp < H - ALR; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance_r = 0.f;

                    __m256 distance_r_vec = _mm256_setzero_ps();
                    __m256 sqdist_vec, color_p_vec, color_q_vec, color_var_p_vec, color_var_q_vec, var_cancel_vec, var_term_vec, normalization_r_vec, dist_var_vec;
                    for (int i=0; i<3; i++){
                        color_p_vec = _mm256_load_ps(color + i * WH + xp * W + yp);
                        color_q_vec = _mm256_loadu_ps(color + i * WH + xq * W + yq);
                        color_var_p_vec = _mm256_load_ps(color_var + i * WH + xp * W + yp);
                        color_var_q_vec = _mm256_loadu_ps(color_var + i * WH + xq * W + yq);

                        sqdist_vec = _mm256_sub_ps(color_p_vec, color_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);

                        var_cancel_vec = _mm256_min_ps(color_var_p_vec, color_var_q_vec);
                        var_cancel_vec = _mm256_add_ps(var_cancel_vec, color_var_p_vec);

                        var_term_vec = _mm256_add_ps(color_var_p_vec, color_var_q_vec);

                        normalization_r_vec = _mm256_mul_ps(k_c_squared_r_vec, var_term_vec);
                        normalization_r_vec = _mm256_add_ps(EPSILON_vec, normalization_r_vec);

                        dist_var_vec = _mm256_sub_ps(sqdist_vec, var_cancel_vec);
                        dist_var_vec = _mm256_div_ps(dist_var_vec, normalization_r_vec);

                        distance_r_vec = _mm256_add_ps(distance_r_vec, dist_var_vec);
                    }

                    _mm256_store_ps(temp+xp*W+yp, distance_r_vec);

                }
            }

            // Precompute feature weights
            for(int xp = ALRfmin; xp < W - ALRfmin; ++xp) {
                for(int yp = ALRfmin; yp < H - ALRfmin; yp+=8) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r = 0.f;
                    scalar df_b = 0.f;
                    __m256 df_r_vec = _mm256_setzero_ps(), df_b_vec = _mm256_setzero_ps();
                    __m256 sqdist_vec, features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
                    __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_r_vec, normalization_b_vec; 
                    __m256 dist_var_vec, tmp_1, tmp_2;
                    for(int j=0; j<NB_FEATURES;++j){
                        features_p_vec = _mm256_load_ps(features+j * WH + xp * W + yp);
                        features_q_vec = _mm256_loadu_ps(features+j * WH + xq * W + yq);
                        features_var_p_vec = _mm256_load_ps(features_var+j * WH + xp * W + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var+j * WH + xq * W + yq);
                        grad_vec = _mm256_load_ps(gradients+j * WH + xp * W + yp);

                        sqdist_vec = _mm256_sub_ps(features_p_vec, features_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);

                        var_cancel_vec = _mm256_min_ps(features_var_q_vec, features_var_p_vec);
                        var_cancel_vec = _mm256_add_ps(features_var_p_vec, var_cancel_vec);
                        var_max_vec = _mm256_max_ps(features_var_p_vec, grad_vec);

                        normalization_r_vec = _mm256_max_ps(tau_r_vec, var_max_vec);
                        normalization_b_vec = _mm256_max_ps(tau_b_vec, var_max_vec);
                        normalization_r_vec = _mm256_mul_ps(k_f_squared_r_vec, normalization_r_vec);
                        normalization_b_vec = _mm256_mul_ps(k_f_squared_b_vec, normalization_b_vec);

                        dist_var_vec = _mm256_sub_ps(sqdist_vec, var_cancel_vec);

                        tmp_1 = _mm256_div_ps(dist_var_vec, normalization_r_vec);
                        tmp_2 = _mm256_div_ps(dist_var_vec, normalization_b_vec);
                        df_r_vec = _mm256_max_ps(df_r_vec, tmp_1);
                        df_b_vec = _mm256_max_ps(df_b_vec, tmp_2);
                        
                    }
                    df_r_vec = _mm256_mul_ps(df_r_vec, _mm256_set1_ps(-1.0));
                    df_b_vec = _mm256_mul_ps(df_b_vec, _mm256_set1_ps(-1.0));
                    __m256 features_weights_r_vec = exp256_ps(df_r_vec);
                    __m256 features_weights_b_vec = exp256_ps(df_b_vec);

                    _mm256_store_ps(features_weights_r+xp * W + yp, features_weights_r_vec);
                    _mm256_store_ps(features_weights_b+xp * W + yp, features_weights_b_vec);
                } 
            }

            
            // Next Steps: Box-Filtering for Patch Contribution 
            // => Use Box-Filter Seperability => linear scans of data
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = ALR; xp < W - ALR; ++xp) {
                for(int yp = ALRr; yp < H - ALRr; yp+=8) {

                    __m256 sum_r_vec = _mm256_setzero_ps();
                    __m256 tmp_vec;
                    for (int k=-f_r; k<=f_r; k++){
                        tmp_vec = _mm256_load_ps(temp+xp * W + yp + k);
                        sum_r_vec = _mm256_add_ps(sum_r_vec, tmp_vec);
                    }
                    _mm256_store_ps(temp2_r+xp * W + yp, sum_r_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = ALRr; xp < W - ALRr; ++xp) {
                for(int yp = ALRr; yp < H - ALRr; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, color_weight_vec, weight_vec, weight_sum_vec;
                    for (int k=-f_r; k<=f_r; k++){
                        tmp_vec = _mm256_load_ps(temp2_r+(xp + k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    color_weight_vec = _mm256_div_ps(sum_vec, neigh_r_vec);
                    color_weight_vec = _mm256_max_ps(color_weight_vec, _mm256_setzero_ps());
                    color_weight_vec = _mm256_mul_ps(color_weight_vec, _mm256_set1_ps(-1.0));
                    color_weight_vec = exp256_ps(color_weight_vec);
                    

                    // Compute final weight
                    weight_vec = _mm256_load_ps(features_weights_r+xp * W + yp);
                    weight_vec = _mm256_min_ps(weight_vec, color_weight_vec);

                    weight_sum_vec = _mm256_load_ps(weight_sum+0 * WH + xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_store_ps(weight_sum+0 * WH + xp * W + yp, weight_sum_vec);
                    
                    __m256 output_r_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_r_vec = _mm256_load_ps(output_r+i * WH + xp * W + yp);
                        color_vec = _mm256_loadu_ps(color+i * WH + xq * W + yq);
                        color_vec = _mm256_mul_ps(color_vec, weight_vec);

                        output_r_vec = _mm256_add_ps(output_r_vec, color_vec);

                        _mm256_store_ps(output_r+i * WH + xp * W + yp, output_r_vec);
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; yp+=8) {
                    
                    __m256 sum_g_vec = _mm256_setzero_ps();
                    __m256 tmp_vec;
                    for (int k=-f_g; k<=f_g; k++){
                        tmp_vec = _mm256_load_ps(temp+xp * W + yp + k);
                        sum_g_vec = _mm256_add_ps(sum_g_vec, tmp_vec);
                    }
                    _mm256_storeu_ps(temp2_g+xp * W + yp, sum_g_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
                for(int yp = R + f_g; yp < H - R - f_g; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, color_weight_vec, weight_vec, weight_sum_vec;
                    for (int k=-f_g; k<=f_g; k++){
                        tmp_vec = _mm256_load_ps(temp2_g+(xp + k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    color_weight_vec = _mm256_div_ps(sum_vec, neigh_g_vec);
                    color_weight_vec = _mm256_max_ps(color_weight_vec, _mm256_setzero_ps());
                    color_weight_vec = _mm256_mul_ps(color_weight_vec, _mm256_set1_ps(-1.0));
                    color_weight_vec = exp256_ps(color_weight_vec);
                    

                    // Compute final weight
                    weight_vec = _mm256_load_ps(features_weights_r+xp * W + yp);
                    weight_vec = _mm256_min_ps(weight_vec, color_weight_vec);

                    weight_sum_vec = _mm256_load_ps(weight_sum+1 * WH + xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum+1 * WH + xp * W + yp, weight_sum_vec);
                    
                    __m256 output_g_vec, color_vec;
                    for (int i=0; i<3; i++){
                        output_g_vec = _mm256_loadu_ps(output_g+i * WH + xp * W + yp);
                        color_vec = _mm256_loadu_ps(color+i * WH + xq * W + yq);
                        color_vec = _mm256_mul_ps(color_vec, weight_vec);

                        output_g_vec = _mm256_add_ps(output_g_vec, color_vec);

                        _mm256_storeu_ps(output_g+i * WH + xp * W + yp, output_g_vec);
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = ALRb; xp < W - ALRb; ++xp) {
                for(int yp = ALRb; yp < H - ALRb; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    __m256 weight_vec, weight_sum_vec, output_b_vec, color_vec;

                    weight_vec = _mm256_load_ps(features_weights_b+xp * W + yp);
                    weight_sum_vec = _mm256_load_ps(weight_sum+2 * WH + xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);                    
                    _mm256_store_ps(weight_sum+2 * WH + xp * W + yp, weight_sum_vec);
                    
                    for (int i=0; i<3; i++){
                        color_vec = _mm256_loadu_ps(color+i * WH + xq * W + yq);
                        color_vec = _mm256_mul_ps(weight_vec, color_vec);
                        output_b_vec = _mm256_load_ps(output_b+i * WH + xp * W + yp);
                        output_b_vec = _mm256_add_ps(output_b_vec, color_vec);
                        _mm256_store_ps(output_b+i * WH + xp * W + yp, output_b_vec);
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = ALRr; xp < W - ALRr; ++xp) {
        for(int yp = ALRr; yp < H - ALRr; yp+=8) {
        
            __m256 weight_sum_vec = _mm256_load_ps(weight_sum+0 * WH + xp * W + yp);
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_load_ps(output_r+i * WH + xp * W + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_store_ps(output_r+i * WH + xp * W + yp, output_vec);
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < H - R - f_g; yp+=8) {
        
            __m256 weight_sum_vec = _mm256_loadu_ps(weight_sum+(1*WH + xp * W + yp));
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output_g+i * WH + xp * W + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_storeu_ps(output_g+i * WH + xp * W + yp, output_vec);
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = ALRb; xp < W - ALRb; ++xp) {
        for(int yp = ALRb; yp < H - ALRb; yp+=8) {
        
            __m256 weight_sum_vec = _mm256_load_ps(weight_sum+(2*WH + xp * W + yp));
            __m256 output_vec;
            for (int i=0; i<3; i++){
                output_vec = _mm256_load_ps(output_b+i * WH + xp * W + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                _mm256_store_ps(output_b+i * WH + xp * W + yp, output_vec);
            }
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + f_r; yp++){
                output_r[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output_r[i * WH + xp * W + H - yp - 1] = color[i * WH + xp * W + H - yp - 1];
                output_b[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output_b[i * WH + xp * W + H - yp - 1] = color[i * WH + xp * W + H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_r; xp++){
            for (int yp = R + f_r ; yp < H - R - f_r; yp++){
            
                output_r[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output_r[i * WH + (W - xp - 1) * W + yp ] = color[i * WH + (W - xp - 1) * W + yp ];
                output_b[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output_b[i * WH + (W - xp - 1) * W + yp ] = color[i * WH + (W - xp - 1) * W + yp ];
             }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + f_g; yp++){
                output_g[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output_g[i * WH + xp * W + H - yp - 1] = color[i * WH + xp * W + H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_g; xp++){
            for (int yp = R + f_g ; yp < H - R - f_g; yp++){
                output_g[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output_g[i * WH + (W - xp - 1) * W + yp ] = color[i * WH + (W - xp - 1) * W + yp ];
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

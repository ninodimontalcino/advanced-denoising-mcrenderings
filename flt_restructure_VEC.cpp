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


void sure_all_VEC(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H){
    
    scalar d_r, d_g, d_b, v;
    __m256 d_r_vec, d_g_vec, d_b_vec, v_vec, c_vec;
    

    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y+=8){
                
            c_vec = _mm256_loadu_ps(c[0][x]+y);
            d_r_vec = _mm256_loadu_ps(cand_r[0][x]+y);
            d_g_vec = _mm256_loadu_ps(cand_g[0][x]+y);
            d_b_vec = _mm256_loadu_ps(cand_b[0][x]+y);
            v_vec = _mm256_loadu_ps(c_var[0][x]+y);

            // d_r = d_r - c
            d_r_vec = _mm256_sub_ps(d_r_vec, c_vec);
            d_g_vec = _mm256_sub_ps(d_g_vec, c_vec);
            d_b_vec = _mm256_sub_ps(d_b_vec, c_vec);

            // Squared
            v_vec = _mm256_mul_ps(v_vec, v_vec);

            // Squared and Difference d_r = d_r-v (use FMA)
            d_r_vec = _mm256_fmsub_ps(d_r_vec, d_r_vec, v_vec);
            d_g_vec = _mm256_fmsub_ps(d_g_vec, d_g_vec, v_vec);
            d_b_vec = _mm256_fmsub_ps(d_b_vec, d_b_vec, v_vec);
                
            _mm256_storeu_ps(sure[0][x]+y, d_r_vec);
            _mm256_storeu_ps(sure[1][x]+y, d_g_vec);
            _mm256_storeu_ps(sure[2][x]+y, d_b_vec);
        }
    }



    for (int i = 1; i < 3; i++){ 
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y+=8){

                // Loading
                c_vec = _mm256_loadu_ps(c[i][x]+y);
                d_r_vec = _mm256_loadu_ps(cand_r[i][x]+y);
                d_g_vec = _mm256_loadu_ps(cand_g[i][x]+y);
                d_b_vec = _mm256_loadu_ps(cand_b[i][x]+y);
                v_vec = _mm256_loadu_ps(c_var[i][x]+y);

                // d_r = d_r - c
                d_r_vec = _mm256_sub_ps(d_r_vec, c_vec);
                d_g_vec = _mm256_sub_ps(d_g_vec, c_vec);
                d_b_vec = _mm256_sub_ps(d_b_vec, c_vec);

                // Squared
                v_vec = _mm256_mul_ps(v_vec, v_vec);
                
                // Difference d_r = d_r-v
                d_r_vec = _mm256_fmsub_ps(d_r_vec, d_r_vec, v_vec);
                d_g_vec = _mm256_fmsub_ps(d_g_vec, d_g_vec, v_vec);
                d_b_vec = _mm256_fmsub_ps(d_b_vec, d_b_vec, v_vec);

                __m256 sure_r_vec = _mm256_loadu_ps(sure[0][x]+y);
                __m256 sure_g_vec = _mm256_loadu_ps(sure[1][x]+y);
                __m256 sure_b_vec = _mm256_loadu_ps(sure[2][x]+y);

                // Summing up
                sure_r_vec = _mm256_add_ps(sure_r_vec, d_r_vec);
                sure_g_vec = _mm256_add_ps(sure_g_vec, d_g_vec);
                sure_b_vec = _mm256_add_ps(sure_b_vec, d_b_vec);
                    
                _mm256_storeu_ps(sure[0][x]+y, sure_r_vec);
                _mm256_storeu_ps(sure[1][x]+y, sure_g_vec);
                _mm256_storeu_ps(sure[2][x]+y, sure_b_vec);
            }
        }
    }
}


void filtering_basic_VEC(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    

    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(k_c_squared);

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W*H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W*H*sizeof(scalar));
    temp2 = (scalar*) malloc(W*H*sizeof(scalar));

    // The end of the vectorized part
    int final_yp = H - 32 + p.r+p.f;


    // Precompute size of neighbourhood
    scalar neigh_inv = 1./ (3*(2*p.f+1)*(2*p.f+1));
    scalar neigh_inv_min = -neigh_inv;
    const __m256 neigh_inv_vec = _mm256_set1_ps(neigh_inv);
    const __m256 neigh_inv_min_vec = _mm256_set1_ps(neigh_inv_min);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, var_cancel_vec, normalization_vec, temp_vec;
            for (int i=0; i<3; i++){  
                for(int xp = p.r; xp < W - p.r; ++xp) {
                    for(int yp = p.r; yp < H - p.r; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp+xp*W+yp);
                        c_var_p_vec = _mm256_loadu_ps(c_var[i][xp]+yp);
                        c_var_q_vec = _mm256_loadu_ps(c_var[i][xq]+yq);
                        c_p_vec = _mm256_loadu_ps(c[i][xp]+yp);
                        c_q_vec = _mm256_loadu_ps(c[i][xq]+yq);

                        normalization_vec = _mm256_add_ps(c_var_p_vec, c_var_q_vec);
                        sqdist_vec = _mm256_sub_ps(c_p_vec, c_q_vec);
                        var_cancel_vec = _mm256_min_ps(c_var_p_vec, c_var_q_vec);

                        normalization_vec = _mm256_mul_ps(k_c_squared_vec, normalization_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_add_ps(c_var_p_vec, var_cancel_vec);

                        normalization_vec = _mm256_add_ps(EPSILON_vec, normalization_vec);

                        sqdist_vec = _mm256_sub_ps(sqdist_vec, var_cancel_vec);
                        sqdist_vec = _mm256_div_ps(sqdist_vec, normalization_vec);

                        temp_vec = _mm256_add_ps(temp_vec, sqdist_vec);
                        _mm256_storeu_ps(temp+xp*W+yp, temp_vec);
                    }
                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            // TODO: maybe this part would be better if not vectorized (to check)
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=64) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();
                    __m256 sum_4_vec = _mm256_setzero_ps();
                    __m256 sum_5_vec = _mm256_setzero_ps();
                    __m256 sum_6_vec = _mm256_setzero_ps();
                    __m256 sum_7_vec = _mm256_setzero_ps();

                    for (int k=-p.f; k<=p.f; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp+xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp+xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp+xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp+xp * W + (yp+3*8)+k);
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp+xp * W + (yp+4*8)+k);
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp+xp * W + (yp+5*8)+k);
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp+xp * W + (yp+6*8)+k);
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp+xp * W + (yp+7*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2+ xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 3 * 8), sum_3_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 4 * 8), sum_4_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 5 * 8), sum_5_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 6 * 8), sum_6_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 7 * 8), sum_7_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-p.f; k<=p.f; k++){
                        __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+0*8));
                        __m256 temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+1*8));
                        __m256 temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+2*8));
                        __m256 temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+3*8));
                        
                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    }

                    // New optimization: using the opposite of neigh_inv to avoid the multiplication by -1, and take the min instead of max
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, neigh_inv_min_vec);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, neigh_inv_min_vec);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, neigh_inv_min_vec);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, neigh_inv_min_vec);

                    weight_vec_0 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    
                    for (int i=0; i<3; i++){
                        __m256 output_vec_0 = _mm256_loadu_ps(output[i][xp] + (yp + 0 * 8));
                        __m256 input_vec_0 = _mm256_loadu_ps(input[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(input[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(input[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(input[i][xq] + (yq + 3 * 8));

                        output_vec_0 = _mm256_fmadd_ps(weight_vec_0, input_vec_0, output_vec_0);
                        output_vec_1 = _mm256_fmadd_ps(weight_vec_1, input_vec_1, output_vec_1);
                        output_vec_2 = _mm256_fmadd_ps(weight_vec_2, input_vec_2, output_vec_2);
                        output_vec_3 = _mm256_fmadd_ps(weight_vec_3, input_vec_3, output_vec_3);

                        _mm256_storeu_ps(output[i][xp] + (yp + 0 * 8), output_vec_0);
                        _mm256_storeu_ps(output[i][xp] + (yp + 1 * 8), output_vec_1);
                        _mm256_storeu_ps(output[i][xp] + (yp + 2 * 8), output_vec_2);
                        _mm256_storeu_ps(output[i][xp] + (yp + 3 * 8), output_vec_3);

                    }
                }

                for(int yp = final_yp; yp < H - p.r - p.f; yp+=4) {

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

                    // New optimization: using the opposite of neigh_inv to avoid the multiplication by -1, and take the min instead of max
                    
                    scalar weight_0 = exp(fmin(0.f, (sum_0 * neigh_inv_min)));
                    scalar weight_1 = exp(fmin(0.f, (sum_1 * neigh_inv_min)));
                    scalar weight_2 = exp(fmin(0.f, (sum_2 * neigh_inv_min)));
                    scalar weight_3 = exp(fmin(0.f, (sum_3 * neigh_inv_min)));
                    
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
        for(int yp = p.r + p.f ; yp < H - 32; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 3 * 8));


            for (int i=0; i<3; i++){
                __m256 output_vec_0 = _mm256_loadu_ps(output[i][xp] + (yp + 0 * 8));
                __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                
                output_vec_0 = _mm256_div_ps(output_vec_0, weight_sum_vec_0);
                output_vec_1 = _mm256_div_ps(output_vec_1, weight_sum_vec_1);
                output_vec_2 = _mm256_div_ps(output_vec_2, weight_sum_vec_2);
                output_vec_3 = _mm256_div_ps(output_vec_3, weight_sum_vec_3);

                _mm256_storeu_ps(output[i][xp] + (yp + 0 * 8), output_vec_0);
                _mm256_storeu_ps(output[i][xp] + (yp + 1 * 8), output_vec_1);
                _mm256_storeu_ps(output[i][xp] + (yp + 2 * 8), output_vec_2);
                _mm256_storeu_ps(output[i][xp] + (yp + 3 * 8), output_vec_3);
            }
        }

        for(int yp = final_yp; yp < H - p.r - p.f; yp+=4) {     
            
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


void feature_prefiltering_VEC(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(k_c_squared);

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
    scalar neigh_inv_min = -neigh_inv;
    const __m256 neigh_inv_vec = _mm256_set1_ps(neigh_inv);
    const __m256 neigh_inv_min_vec = _mm256_set1_ps(neigh_inv_min);

    // The end of the vectorized part
    int final_yp = H - 32 + p.r+p.f;

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){
                for(int xp = p.r; xp < W - p.r; ++xp) {
                    for(int yp = p.r; yp < H - p.r; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        __m256 sqdist_vec, normalization_vec, var_cancel_vec;
                        __m256 temp_vec = _mm256_loadu_ps(temp+ xp * W + yp);
                        __m256 features_p_vec = _mm256_loadu_ps(features[i][xp]+yp);
                        __m256 features_q_vec = _mm256_loadu_ps(features[i][xq]+yq);
                        __m256 features_var_p_vec = _mm256_loadu_ps(features_var[i][xp]+yp);
                        __m256 features_var_q_vec = _mm256_loadu_ps(features_var[i][xq]+yq);

                        normalization_vec = _mm256_add_ps(features_var_p_vec, features_var_q_vec);
                        sqdist_vec = _mm256_sub_ps(features_p_vec, features_q_vec);
                        var_cancel_vec = _mm256_min_ps(features_var_p_vec, features_var_q_vec);

                        normalization_vec = _mm256_mul_ps(k_c_squared_vec, normalization_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_add_ps(features_var_p_vec, var_cancel_vec);

                        normalization_vec = _mm256_add_ps(EPSILON_vec, normalization_vec);

                        sqdist_vec = _mm256_sub_ps(sqdist_vec, var_cancel_vec);
                        sqdist_vec = _mm256_div_ps(sqdist_vec, normalization_vec);

                        temp_vec = _mm256_add_ps(temp_vec, sqdist_vec);
                        _mm256_storeu_ps(temp+xp*W+yp, temp_vec);
                        
                    }
                }
            }
            

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=64) {
                    
                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();
                    __m256 sum_4_vec = _mm256_setzero_ps();
                    __m256 sum_5_vec = _mm256_setzero_ps();
                    __m256 sum_6_vec = _mm256_setzero_ps();
                    __m256 sum_7_vec = _mm256_setzero_ps();

                    for (int k=-p.f; k<=p.f; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp+xp * W + yp+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp+xp * W + (yp+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp+xp * W + (yp+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp+xp * W + (yp+3*8)+k);
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp+xp * W + (yp+4*8)+k);
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp+xp * W + (yp+5*8)+k);
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp+xp * W + (yp+6*8)+k);
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp+xp * W + (yp+7*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2+ xp * W + (yp + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 3 * 8), sum_3_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 4 * 8), sum_4_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 5 * 8), sum_5_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 6 * 8), sum_6_vec);
                    _mm256_storeu_ps(temp2+ xp * W + (yp + 7 * 8), sum_7_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-p.f; k<=p.f; k++){
                        __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+0*8));
                        __m256 temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+1*8));
                        __m256 temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+2*8));
                        __m256 temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+k) * W + (yp+3*8));
                        
                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    }

                    // New optimization: using the opposite of neigh_inv to avoid the multiplication by -1, and take the min instead of max
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, neigh_inv_min_vec);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, neigh_inv_min_vec);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, neigh_inv_min_vec);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, neigh_inv_min_vec);

                    weight_vec_0 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(_mm256_setzero_ps(), weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum+ xp * W + ( yp + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum+ xp * W + ( yp + 3 * 8), weight_sum_vec_3);
                    
                    
                    for (int i=0; i<3; i++){
                        __m256 output_vec_0 = _mm256_loadu_ps(output[i][xp] + (yp + 0 * 8));
                        __m256 features_vec_0 = _mm256_loadu_ps(features[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                        __m256 features_vec_1 = _mm256_loadu_ps(features[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                        __m256 features_vec_2 = _mm256_loadu_ps(features[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                        __m256 features_vec_3 = _mm256_loadu_ps(features[i][xq] + (yq + 3 * 8));
                        __m256 output_var_vec_0 = _mm256_loadu_ps(output_var[i][xp] + (yp + 0 * 8));
                        __m256 features_var_vec_0 = _mm256_loadu_ps(features_var[i][xq] + (yq + 0 * 8));
                        __m256 output_var_vec_1 = _mm256_loadu_ps(output_var[i][xp] + (yp + 1 * 8));
                        __m256 features_var_vec_1 = _mm256_loadu_ps(features_var[i][xq] + (yq + 1 * 8));
                        __m256 output_var_vec_2 = _mm256_loadu_ps(output_var[i][xp] + (yp + 2 * 8));
                        __m256 features_var_vec_2 = _mm256_loadu_ps(features_var[i][xq] + (yq + 2 * 8));
                        __m256 output_var_vec_3 = _mm256_loadu_ps(output_var[i][xp] + (yp + 3 * 8));
                        __m256 features_var_vec_3 = _mm256_loadu_ps(features_var[i][xq] + (yq + 3 * 8));

                        output_vec_0 = _mm256_fmadd_ps(weight_vec_0, features_vec_0, output_vec_0);
                        output_vec_1 = _mm256_fmadd_ps(weight_vec_1, features_vec_1, output_vec_1);
                        output_vec_2 = _mm256_fmadd_ps(weight_vec_2, features_vec_2, output_vec_2);
                        output_vec_3 = _mm256_fmadd_ps(weight_vec_3, features_vec_3, output_vec_3);

                        output_var_vec_0 = _mm256_fmadd_ps(weight_vec_0, features_var_vec_0, output_var_vec_0);
                        output_var_vec_1 = _mm256_fmadd_ps(weight_vec_1, features_var_vec_1, output_var_vec_1);
                        output_var_vec_2 = _mm256_fmadd_ps(weight_vec_2, features_var_vec_2, output_var_vec_2);
                        output_var_vec_3 = _mm256_fmadd_ps(weight_vec_3, features_var_vec_3, output_var_vec_3);

                        _mm256_storeu_ps(output[i][xp] + (yp + 0 * 8), output_vec_0);
                        _mm256_storeu_ps(output[i][xp] + (yp + 1 * 8), output_vec_1);
                        _mm256_storeu_ps(output[i][xp] + (yp + 2 * 8), output_vec_2);
                        _mm256_storeu_ps(output[i][xp] + (yp + 3 * 8), output_vec_3);
                        _mm256_storeu_ps(output_var[i][xp] + (yp + 0 * 8), output_var_vec_0);
                        _mm256_storeu_ps(output_var[i][xp] + (yp + 1 * 8), output_var_vec_1);
                        _mm256_storeu_ps(output_var[i][xp] + (yp + 2 * 8), output_var_vec_2);
                        _mm256_storeu_ps(output_var[i][xp] + (yp + 3 * 8), output_var_vec_3);

                    }
                }

                for(int yp = final_yp; yp < H - p.r - p.f; yp+=4) {

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

                    scalar weight_0 = exp(fmin(0.f, (sum_0 * neigh_inv_min)));
                    scalar weight_1 = exp(fmin(0.f, (sum_1 * neigh_inv_min)));
                    scalar weight_2 = exp(fmin(0.f, (sum_2 * neigh_inv_min)));
                    scalar weight_3 = exp(fmin(0.f, (sum_3 * neigh_inv_min)));

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
        for(int yp = p.r + p.f ; yp < H - 32; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum + xp * W + (yp + 3 * 8));


            for (int i=0; i<3; i++){
                __m256 output_vec_0 = _mm256_loadu_ps(output[i][xp] + (yp + 0 * 8));
                __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                __m256 output_var_vec_0 = _mm256_loadu_ps(output_var[i][xp] + (yp + 0 * 8));
                __m256 output_var_vec_1 = _mm256_loadu_ps(output_var[i][xp] + (yp + 1 * 8));
                __m256 output_var_vec_2 = _mm256_loadu_ps(output_var[i][xp] + (yp + 2 * 8));
                __m256 output_var_vec_3 = _mm256_loadu_ps(output_var[i][xp] + (yp + 3 * 8));
                
                output_vec_0 = _mm256_div_ps(output_vec_0, weight_sum_vec_0);
                output_vec_1 = _mm256_div_ps(output_vec_1, weight_sum_vec_1);
                output_vec_2 = _mm256_div_ps(output_vec_2, weight_sum_vec_2);
                output_vec_3 = _mm256_div_ps(output_vec_3, weight_sum_vec_3);
                output_var_vec_0 = _mm256_div_ps(output_var_vec_0, weight_sum_vec_0);
                output_var_vec_1 = _mm256_div_ps(output_var_vec_1, weight_sum_vec_1);
                output_var_vec_2 = _mm256_div_ps(output_var_vec_2, weight_sum_vec_2);
                output_var_vec_3 = _mm256_div_ps(output_var_vec_3, weight_sum_vec_3);

                _mm256_storeu_ps(output[i][xp] + (yp + 0 * 8), output_vec_0);
                _mm256_storeu_ps(output[i][xp] + (yp + 1 * 8), output_vec_1);
                _mm256_storeu_ps(output[i][xp] + (yp + 2 * 8), output_vec_2);
                _mm256_storeu_ps(output[i][xp] + (yp + 3 * 8), output_vec_3);
                _mm256_storeu_ps(output_var[i][xp] + (yp + 0 * 8), output_var_vec_0);
                _mm256_storeu_ps(output_var[i][xp] + (yp + 1 * 8), output_var_vec_1);
                _mm256_storeu_ps(output_var[i][xp] + (yp + 2 * 8), output_var_vec_2);
                _mm256_storeu_ps(output_var[i][xp] + (yp + 3 * 8), output_var_vec_3);
            }
        }

        for(int yp = final_yp; yp < H - p.r - p.f; yp+=4) {     
            
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

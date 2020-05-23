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


void sure_all_BLK(buffer sure, buffer color, buffer color_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H){
    
    scalar d_r, d_g, d_b, v;
    __m256 d_r_vec, d_g_vec, d_b_vec, v_vec, c_vec;
    

    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y+=8){
                
            c_vec = _mm256_loadu_ps(color[0][x]+y);
            d_r_vec = _mm256_loadu_ps(cand_r[0][x]+y);
            d_g_vec = _mm256_loadu_ps(cand_g[0][x]+y);
            d_b_vec = _mm256_loadu_ps(cand_b[0][x]+y);
            v_vec = _mm256_loadu_ps(color_var[0][x]+y);

            // d_r = d_r - color
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
                c_vec = _mm256_loadu_ps(color[i][x]+y);
                d_r_vec = _mm256_loadu_ps(cand_r[i][x]+y);
                d_g_vec = _mm256_loadu_ps(cand_g[i][x]+y);
                d_b_vec = _mm256_loadu_ps(cand_b[i][x]+y);
                v_vec = _mm256_loadu_ps(color_var[i][x]+y);

                // d_r = d_r - color
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


void filtering_basic_BLK(buffer output, buffer input, buffer color, buffer color_var, Flt_parameters p, int W, int H){
    

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
                        c_var_p_vec = _mm256_loadu_ps(color_var[i][xp]+yp);
                        c_var_q_vec = _mm256_loadu_ps(color_var[i][xq]+yq);
                        c_p_vec = _mm256_loadu_ps(color[i][xp]+yp);
                        c_q_vec = _mm256_loadu_ps(color[i][xq]+yq);

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



void filtering_basic_f3_BLK(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    scalar K_C_SQUARED = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(K_C_SQUARED);
    int R = p.r;
    int F = p.f;
    
    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar sqdist0, sqdist1, sqdist2; 
    scalar var_cancel0, var_cancel1, var_cancel2;
    scalar normalization0, normalization1, normalization2;
    scalar term0, term1, term2;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;

    __m256 c_p_vec_0, c_q_vec_0, c_var_p_vec_0, c_var_q_vec_0, var_cancel_vec_0;
    __m256 c_p_vec_1, c_q_vec_1, c_var_p_vec_1, c_var_q_vec_1, var_cancel_vec_1;
    __m256 c_p_vec_2, c_q_vec_2, c_var_p_vec_2, c_var_q_vec_2, var_cancel_vec_2;
    __m256 sqdist_vec_0, sqdist_vec_1, sqdist_vec_2;
    __m256 normalization_vec_0, normalization_vec_1, normalization_vec_2;

    // ------------------------
    // MEMORY ALLOCATION
    // ------------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W*H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W*H*sizeof(scalar));
    temp2 = (scalar*) malloc(W*H*sizeof(scalar));

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1./ (3*(2*F+1)*(2*F+1));
    scalar NEIGH_INV_MIN = - NEIGH_INV;
    __m256 NEIGH_INV_MIN_VEC = _mm256_set1_ps(NEIGH_INV_MIN);

    // The end of the vectorized part
    int final_yp = H - 32 + p.r+p.f;

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; yp+=8) {

                    xq = xp + r_x;
                    yq = yp + r_y;
                    
                    c_var_p_vec_0 = _mm256_loadu_ps(c_var[0][xp]+yp);
                    c_var_q_vec_0 = _mm256_loadu_ps(c_var[0][xq]+yq);
                    c_p_vec_0 = _mm256_loadu_ps(c[0][xp]+yp);
                    c_q_vec_0 = _mm256_loadu_ps(c[0][xq]+yq);
                    c_var_p_vec_1 = _mm256_loadu_ps(c_var[1][xp]+yp);
                    c_var_q_vec_1 = _mm256_loadu_ps(c_var[1][xq]+yq);
                    c_p_vec_1 = _mm256_loadu_ps(c[1][xp]+yp);
                    c_q_vec_1 = _mm256_loadu_ps(c[1][xq]+yq);
                    c_var_p_vec_2 = _mm256_loadu_ps(c_var[2][xp]+yp);
                    c_var_q_vec_2 = _mm256_loadu_ps(c_var[2][xq]+yq);
                    c_p_vec_2 = _mm256_loadu_ps(c[2][xp]+yp);
                    c_q_vec_2 = _mm256_loadu_ps(c[2][xq]+yq);

                    normalization_vec_0 = _mm256_add_ps(c_var_p_vec_0, c_var_q_vec_0);
                    normalization_vec_1 = _mm256_add_ps(c_var_p_vec_1, c_var_q_vec_1);
                    normalization_vec_2 = _mm256_add_ps(c_var_p_vec_2, c_var_q_vec_2);
                    sqdist_vec_0 = _mm256_sub_ps(c_p_vec_0, c_q_vec_0);
                    sqdist_vec_1 = _mm256_sub_ps(c_p_vec_1, c_q_vec_1);
                    sqdist_vec_2 = _mm256_sub_ps(c_p_vec_2, c_q_vec_2);
                    var_cancel_vec_0 = _mm256_min_ps(c_var_p_vec_0, c_var_q_vec_0);
                    var_cancel_vec_1 = _mm256_min_ps(c_var_p_vec_1, c_var_q_vec_1);
                    var_cancel_vec_2 = _mm256_min_ps(c_var_p_vec_2, c_var_q_vec_2);

                    normalization_vec_0 = _mm256_mul_ps(k_c_squared_vec, normalization_vec_0);
                    normalization_vec_1 = _mm256_mul_ps(k_c_squared_vec, normalization_vec_1);
                    normalization_vec_2 = _mm256_mul_ps(k_c_squared_vec, normalization_vec_2);
                    sqdist_vec_0 = _mm256_mul_ps(sqdist_vec_0, sqdist_vec_0);
                    sqdist_vec_1 = _mm256_mul_ps(sqdist_vec_1, sqdist_vec_1);
                    sqdist_vec_2 = _mm256_mul_ps(sqdist_vec_2, sqdist_vec_2);
                    var_cancel_vec_0 = _mm256_add_ps(c_var_p_vec_0, var_cancel_vec_0);
                    var_cancel_vec_1 = _mm256_add_ps(c_var_p_vec_1, var_cancel_vec_1);
                    var_cancel_vec_2 = _mm256_add_ps(c_var_p_vec_2, var_cancel_vec_2);

                    normalization_vec_0 = _mm256_add_ps(EPSILON_vec, normalization_vec_0);
                    sqdist_vec_0 = _mm256_sub_ps(sqdist_vec_0, var_cancel_vec_0);
                    normalization_vec_1 = _mm256_add_ps(EPSILON_vec, normalization_vec_1);
                    sqdist_vec_1 = _mm256_sub_ps(sqdist_vec_1, var_cancel_vec_1);
                    normalization_vec_2 = _mm256_add_ps(EPSILON_vec, normalization_vec_2);
                    sqdist_vec_2 = _mm256_sub_ps(sqdist_vec_2, var_cancel_vec_2);

                    sqdist_vec_0 = _mm256_div_ps(sqdist_vec_0, normalization_vec_0);
                    sqdist_vec_1 = _mm256_div_ps(sqdist_vec_1, normalization_vec_1);
                    sqdist_vec_2 = _mm256_div_ps(sqdist_vec_2, normalization_vec_2);

                    sqdist_vec_0 = _mm256_add_ps(sqdist_vec_0, sqdist_vec_1);
                    sqdist_vec_0 = _mm256_add_ps(sqdist_vec_0, sqdist_vec_2);

                    _mm256_storeu_ps(temp+xp*W+yp, sqdist_vec_0);
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = p.r + p.f; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+0*8));
                    __m256 sum_1_vec = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+1*8));
                    __m256 sum_2_vec = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+2*8));
                    __m256 sum_3_vec = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+3*8));

                    __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+0*8));
                    __m256 temp2_vec_1 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+1*8));
                    __m256 temp2_vec_2 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+2*8));
                    __m256 temp2_vec_3 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);
                    

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);


                    temp2_vec_0 = _mm256_loadu_ps(temp2+xp * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+xp * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+xp * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+xp * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);


                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    // New optimization: using the opposite of neigh_inv to avoid the multiplication by -1, and take the min instead of max
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, NEIGH_INV_MIN_VEC);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, NEIGH_INV_MIN_VEC);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, NEIGH_INV_MIN_VEC);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, NEIGH_INV_MIN_VEC);

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

                    sum_0 += temp2[(xp-3) * W + yp];
                    sum_1 += temp2[(xp-3) * W + yp+1];
                    sum_2 += temp2[(xp-3) * W + yp+2];
                    sum_3 += temp2[(xp-3) * W + yp+3];

                    sum_0 += temp2[(xp-2) * W + yp];
                    sum_1 += temp2[(xp-2) * W + yp+1];
                    sum_2 += temp2[(xp-2) * W + yp+2];
                    sum_3 += temp2[(xp-2) * W + yp+3];

                    sum_0 += temp2[(xp-1) * W + yp];
                    sum_1 += temp2[(xp-1) * W + yp+1];
                    sum_2 += temp2[(xp-1) * W + yp+2];
                    sum_3 += temp2[(xp-1) * W + yp+3];

                    sum_0 += temp2[(xp) * W + yp];
                    sum_1 += temp2[(xp) * W + yp+1];
                    sum_2 += temp2[(xp) * W + yp+2];
                    sum_3 += temp2[(xp) * W + yp+3];

                    sum_0 += temp2[(xp+1) * W + yp];
                    sum_1 += temp2[(xp+1) * W + yp+1];
                    sum_2 += temp2[(xp+1) * W + yp+2];
                    sum_3 += temp2[(xp+1) * W + yp+3];

                    sum_0 += temp2[(xp+2) * W + yp];
                    sum_1 += temp2[(xp+2) * W + yp+1];
                    sum_2 += temp2[(xp+2) * W + yp+2];
                    sum_3 += temp2[(xp+2) * W + yp+3];

                    sum_0 += temp2[(xp+3) * W + yp];
                    sum_1 += temp2[(xp+3) * W + yp+1];
                    sum_2 += temp2[(xp+3) * W + yp+2];
                    sum_3 += temp2[(xp+3) * W + yp+3];
                    
                    // Final Weight
                    weight_0 = exp(-fmax(0.f, (sum_0 * NEIGH_INV)));
                    weight_1 = exp(-fmax(0.f, (sum_1 * NEIGH_INV)));
                    weight_2 = exp(-fmax(0.f, (sum_2 * NEIGH_INV)));
                    weight_3 = exp(-fmax(0.f, (sum_3 * NEIGH_INV)));
                    
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
    for(int xp = R + F; xp < W - R - F; ++xp) {
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

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][H - yp - 1] = input[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F; xp++){
            for (int yp = R + F ; yp < H - R - F; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][W - xp - 1][yp] = input[i][W - xp - 1][yp];
            }
        }
    }


    free(weight_sum);
    free(temp);
    free(temp2); 
}

void filtering_basic_f1_BLK(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    scalar K_C_SQUARED = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(K_C_SQUARED);
    int R = p.r;
    int F = p.f;
    
    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar sqdist0, sqdist1, sqdist2; 
    scalar var_cancel0, var_cancel1, var_cancel2;
    scalar normalization0, normalization1, normalization2;
    scalar term0, term1, term2;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;

    __m256 c_p_vec_0, c_q_vec_0, c_var_p_vec_0, c_var_q_vec_0, var_cancel_vec_0;
    __m256 c_p_vec_1, c_q_vec_1, c_var_p_vec_1, c_var_q_vec_1, var_cancel_vec_1;
    __m256 c_p_vec_2, c_q_vec_2, c_var_p_vec_2, c_var_q_vec_2, var_cancel_vec_2;
    __m256 sqdist_vec_0, sqdist_vec_1, sqdist_vec_2;
    __m256 normalization_vec_0, normalization_vec_1, normalization_vec_2;

    // ------------------------
    // MEMORY ALLOCATION
    // ------------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W*H, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    temp = (scalar*) malloc(W*H*sizeof(scalar));
    temp2 = (scalar*) malloc(W*H*sizeof(scalar));

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1./ (3*(2*F+1)*(2*F+1));
    scalar NEIGH_INV_MIN = - NEIGH_INV;
    __m256 NEIGH_INV_MIN_VEC = _mm256_set1_ps(NEIGH_INV_MIN);


    // The end of the vectorized part
    int final_yp = H - 32 + p.r+p.f;

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; yp+=8) {

                    xq = xp + r_x;
                    yq = yp + r_y;
                    
                    sqdist0 = c[0][xp][yp] - c[0][xq][yq];
                    sqdist1 = c[1][xp][yp] - c[1][xq][yq];
                    sqdist2 = c[2][xp][yp] - c[2][xq][yq];

                    sqdist0 *= sqdist0;
                    sqdist1 *= sqdist1;
                    sqdist2 *= sqdist2;

                    var_cancel0 = c_var[0][xp][yp] + fmin(c_var[0][xp][yp], c_var[0][xq][yq]);
                    var_cancel1 = c_var[1][xp][yp] + fmin(c_var[1][xp][yp], c_var[1][xq][yq]);
                    var_cancel2 = c_var[2][xp][yp] + fmin(c_var[2][xp][yp], c_var[2][xq][yq]);

                    normalization0 = EPSILON + K_C_SQUARED*(c_var[0][xp][yp] + c_var[0][xq][yq]);
                    normalization1 = EPSILON + K_C_SQUARED*(c_var[1][xp][yp] + c_var[1][xq][yq]);
                    normalization2 = EPSILON + K_C_SQUARED*(c_var[2][xp][yp] + c_var[2][xq][yq]);
                    
                    term0 = (sqdist0 - var_cancel0) / normalization0;
                    term1 = (sqdist1 - var_cancel1) / normalization1;
                    term2 = (sqdist2 - var_cancel2) / normalization2;
                    
                    temp[xp * W + yp] = term0 + term1 + term2;

                    c_var_p_vec_0 = _mm256_loadu_ps(c_var[0][xp]+yp);
                    c_var_q_vec_0 = _mm256_loadu_ps(c_var[0][xq]+yq);
                    c_p_vec_0 = _mm256_loadu_ps(c[0][xp]+yp);
                    c_q_vec_0 = _mm256_loadu_ps(c[0][xq]+yq);
                    c_var_p_vec_1 = _mm256_loadu_ps(c_var[1][xp]+yp);
                    c_var_q_vec_1 = _mm256_loadu_ps(c_var[1][xq]+yq);
                    c_p_vec_1 = _mm256_loadu_ps(c[1][xp]+yp);
                    c_q_vec_1 = _mm256_loadu_ps(c[1][xq]+yq);
                    c_var_p_vec_2 = _mm256_loadu_ps(c_var[2][xp]+yp);
                    c_var_q_vec_2 = _mm256_loadu_ps(c_var[2][xq]+yq);
                    c_p_vec_2 = _mm256_loadu_ps(c[2][xp]+yp);
                    c_q_vec_2 = _mm256_loadu_ps(c[2][xq]+yq);

                    normalization_vec_0 = _mm256_add_ps(c_var_p_vec_0, c_var_q_vec_0);
                    normalization_vec_1 = _mm256_add_ps(c_var_p_vec_1, c_var_q_vec_1);
                    normalization_vec_2 = _mm256_add_ps(c_var_p_vec_2, c_var_q_vec_2);
                    sqdist_vec_0 = _mm256_sub_ps(c_p_vec_0, c_q_vec_0);
                    sqdist_vec_1 = _mm256_sub_ps(c_p_vec_1, c_q_vec_1);
                    sqdist_vec_2 = _mm256_sub_ps(c_p_vec_2, c_q_vec_2);
                    var_cancel_vec_0 = _mm256_min_ps(c_var_p_vec_0, c_var_q_vec_0);
                    var_cancel_vec_1 = _mm256_min_ps(c_var_p_vec_1, c_var_q_vec_1);
                    var_cancel_vec_2 = _mm256_min_ps(c_var_p_vec_2, c_var_q_vec_2);

                    normalization_vec_0 = _mm256_mul_ps(k_c_squared_vec, normalization_vec_0);
                    normalization_vec_1 = _mm256_mul_ps(k_c_squared_vec, normalization_vec_1);
                    normalization_vec_2 = _mm256_mul_ps(k_c_squared_vec, normalization_vec_2);
                    sqdist_vec_0 = _mm256_mul_ps(sqdist_vec_0, sqdist_vec_0);
                    sqdist_vec_1 = _mm256_mul_ps(sqdist_vec_1, sqdist_vec_1);
                    sqdist_vec_2 = _mm256_mul_ps(sqdist_vec_2, sqdist_vec_2);
                    var_cancel_vec_0 = _mm256_add_ps(c_var_p_vec_0, var_cancel_vec_0);
                    var_cancel_vec_1 = _mm256_add_ps(c_var_p_vec_1, var_cancel_vec_1);
                    var_cancel_vec_2 = _mm256_add_ps(c_var_p_vec_2, var_cancel_vec_2);

                    normalization_vec_0 = _mm256_add_ps(EPSILON_vec, normalization_vec_0);
                    sqdist_vec_0 = _mm256_sub_ps(sqdist_vec_0, var_cancel_vec_0);
                    normalization_vec_1 = _mm256_add_ps(EPSILON_vec, normalization_vec_1);
                    sqdist_vec_1 = _mm256_sub_ps(sqdist_vec_1, var_cancel_vec_1);
                    normalization_vec_2 = _mm256_add_ps(EPSILON_vec, normalization_vec_2);
                    sqdist_vec_2 = _mm256_sub_ps(sqdist_vec_2, var_cancel_vec_2);

                    sqdist_vec_0 = _mm256_div_ps(sqdist_vec_0, normalization_vec_0);
                    sqdist_vec_1 = _mm256_div_ps(sqdist_vec_1, normalization_vec_1);
                    sqdist_vec_2 = _mm256_div_ps(sqdist_vec_2, normalization_vec_2);

                    sqdist_vec_0 = _mm256_add_ps(sqdist_vec_0, sqdist_vec_1);
                    sqdist_vec_0 = _mm256_add_ps(sqdist_vec_0, sqdist_vec_2);

                    _mm256_storeu_ps(temp+xp*W+yp, sqdist_vec_0);
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
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
            for(int xp = R + F; xp < W - R - F; ++xp) {                
                for(int yp = p.r + p.f; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+0*8));
                    __m256 sum_1_vec = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+1*8));
                    __m256 sum_2_vec = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+2*8));
                    __m256 sum_3_vec = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+3*8));


                    __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+xp * W + (yp+0*8));
                    __m256 temp2_vec_1 = _mm256_loadu_ps(temp2+xp * W + (yp+1*8));
                    __m256 temp2_vec_2 = _mm256_loadu_ps(temp2+xp * W + (yp+2*8));
                    __m256 temp2_vec_3 = _mm256_loadu_ps(temp2+xp * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);


                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+3*8));
                        
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    // New optimization: using the opposite of neigh_inv to avoid the multiplication by -1, and take the min instead of max
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, NEIGH_INV_MIN_VEC);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, NEIGH_INV_MIN_VEC);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, NEIGH_INV_MIN_VEC);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, NEIGH_INV_MIN_VEC);

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

                    // Unrolled summation => tailed to f=1 => 2*f+1 = 3
                    sum_0 += temp2[(xp-1) * W + yp];
                    sum_1 += temp2[(xp-1) * W + yp+1];
                    sum_2 += temp2[(xp-1) * W + yp+2];
                    sum_3 += temp2[(xp-1) * W + yp+3];

                    sum_0 += temp2[(xp) * W + yp];
                    sum_1 += temp2[(xp) * W + yp+1];
                    sum_2 += temp2[(xp) * W + yp+2];
                    sum_3 += temp2[(xp) * W + yp+3];

                    sum_0 += temp2[(xp+1) * W + yp];
                    sum_1 += temp2[(xp+1) * W + yp+1];
                    sum_2 += temp2[(xp+1) * W + yp+2];
                    sum_3 += temp2[(xp+1) * W + yp+3];
                    
                    
                    weight_0 = exp(fmin(0.f, (sum_0 * NEIGH_INV_MIN)));
                    weight_1 = exp(fmin(0.f, (sum_1 * NEIGH_INV_MIN)));
                    weight_2 = exp(fmin(0.f, (sum_2 * NEIGH_INV_MIN)));
                    weight_3 = exp(fmin(0.f, (sum_3 * NEIGH_INV_MIN)));
                    
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
    for(int xp = R + F; xp < W - R - F; ++xp) {
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

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][H - yp - 1] = input[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F; xp++){
            for (int yp = R + F ; yp < H - R - F; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][W - xp - 1][yp] = input[i][W - xp - 1][yp];
            }
        }
    }


    free(weight_sum);
    free(temp);
    free(temp2); 
}



void feature_prefiltering_BLK(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int X0, int Y0, int W, int H){

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
    const int tempW = W + 2*p.f + 1;
    const int tempH = H + 2*p.f + 1;
    temp = (scalar*) malloc(tempW*tempH*sizeof(scalar));
    temp2 = (scalar*) malloc(tempW*tempH*sizeof(scalar));
    // printf("size: %d\n",tempW*tempH);

    // Precompute size of neighbourhood
    scalar neigh_inv = 1. / (3*(2*p.f+1)*(2*p.f+1));
    scalar neigh_inv_min = -neigh_inv;
    const __m256 neigh_inv_vec = _mm256_set1_ps(neigh_inv);
    const __m256 neigh_inv_min_vec = _mm256_set1_ps(neigh_inv_min);

    // The end of the vectorized part
    int final_yp = H + Y0 - 32;

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, tempW*tempH*sizeof(scalar));
            for (int i=0; i<3; i++){
                for (int xp =  X0-p.f; xp < W + X0 + p.f; ++xp){
                    for(int yp = Y0-p.f; yp < H + Y0 + p.f; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        __m256 sqdist_vec, normalization_vec, var_cancel_vec;
                        __m256 temp_vec = _mm256_loadu_ps(temp+ (xp-X0 + p.f) * tempW + (yp - Y0 + p.f));
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
                        _mm256_storeu_ps(temp+(xp-X0+p.f)*tempW+(yp-Y0+p.f), temp_vec);
                        
                    }
                }
            }
            

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for (int xp =  X0-p.f; xp < X0+W+p.f; ++xp){
                for(int yp = Y0; yp < H +Y0; yp+=64) {
                    
                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();
                    __m256 sum_4_vec = _mm256_setzero_ps();
                    __m256 sum_5_vec = _mm256_setzero_ps();
                    __m256 sum_6_vec = _mm256_setzero_ps();
                    __m256 sum_7_vec = _mm256_setzero_ps();

                    for (int k=-p.f; k<=p.f; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + (yp-Y0)+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+3*8)+k);
                        __m256 temp_vec_4 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+4*8)+k);
                        __m256 temp_vec_5 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+5*8)+k);
                        __m256 temp_vec_6 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+6*8)+k);
                        __m256 temp_vec_7 = _mm256_loadu_ps(temp+(xp-X0+p.f) * tempW + ((yp-Y0)+7*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                        sum_4_vec = _mm256_add_ps(sum_4_vec, temp_vec_4);
                        sum_5_vec = _mm256_add_ps(sum_5_vec, temp_vec_5);
                        sum_6_vec = _mm256_add_ps(sum_6_vec, temp_vec_6);
                        sum_7_vec = _mm256_add_ps(sum_7_vec, temp_vec_7);
                    }

                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 3 * 8), sum_3_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 4 * 8), sum_4_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 5 * 8), sum_5_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 6 * 8), sum_6_vec);
                    _mm256_storeu_ps(temp2+ (xp-X0+p.f) * tempW + ((yp-Y0) + 7 * 8), sum_7_vec);
                //     if (r_x == 0 && r_y == 0)
                //         printf("Access: %d\n", (xp-X0+p.f) * tempW + ((yp-Y0) + 7 * 8));
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = X0; xp < W+X0; ++xp) {
                for(int yp = Y0; yp < H+Y0; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-p.f; k<=p.f; k++){
                        __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+((xp-X0)+k) * tempW + ((yp-Y0)+0*8));
                        __m256 temp2_vec_1 = _mm256_loadu_ps(temp2+((xp-X0)+k) * tempW + ((yp-Y0)+1*8));
                        __m256 temp2_vec_2 = _mm256_loadu_ps(temp2+((xp-X0)+k) * tempW + ((yp-Y0)+2*8));
                        __m256 temp2_vec_3 = _mm256_loadu_ps(temp2+((xp-X0)+k) * tempW + ((yp-Y0)+3*8));
                        
                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    }
                    // if (r_x == 0 && r_y == 0)
                    //     printf("Access: %d\n", (xp-X0+p.f) * tempW + ((yp-Y0) + 3 * 8));


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

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum+ (xp-X0) * W + ( (yp-Y0) + 3 * 8), weight_sum_vec_3);
                    
                    
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
            }
        }
    }

    // Final Weight Normalization
    for(int xp = X0; xp < W + X0; ++xp) {
        for(int yp = Y0 ; yp < H + Y0; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum + (xp-X0) * W + ((yp-Y0) + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum + (xp-X0) * W + ((yp-Y0) + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum + (xp-X0) * W + ((yp-Y0) + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum + (xp-X0) * W + ((yp-Y0) + 3 * 8));


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
    }

    // Free memory
    free(weight_sum);
    free(temp);
    free(temp2); 

}


void candidate_filtering_FIRST_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    scalar K_C_SQUARED = p.kc * p.kc;
    scalar K_F_SQUARED = p.kf * p.kf;
    scalar tau = p.tau;
    int R = p.r;
    int F = p.f;
    const __m256 k_c_squared_vec = _mm256_set1_ps(K_C_SQUARED);
    const __m256 k_f_squared_vec = _mm256_set1_ps(K_F_SQUARED);
    const __m256 tau_vec = _mm256_set1_ps(tau);

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));
    const __m256 neigh_inv_vec = _mm256_set1_ps(NEIGH_INV);


    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar sqdist, var_cancel, dist_var, var_term, var_max, normalization, df;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;
    scalar final_yp = H - 32 + R + F;

    // ------------------------
    // MEMORY ALLOCATION
    // ------------------------

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
    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1, tmp_2, grad_vec;
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+F; x < W - R - F; ++x) {
            for(int y =  R+F; y < H -  R - F; y+=8) {
                
                // Loading
                features_vec  = _mm256_loadu_ps(features[i][x] + y);
                diffL_sqr_vec = _mm256_loadu_ps(features[i][x-1] + y);
                diffR_sqr_vec =_mm256_loadu_ps(features[i][x+1]+ y);
                diffU_sqr_vec =_mm256_loadu_ps(features[i][x] + y-1);
                diffD_sqr_vec =_mm256_loadu_ps(features[i][x] + y+1);

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

                _mm256_storeu_ps(gradients+i * WH + x * W + y, tmp_1);

            } 
        }
    }

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){

            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));

            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight;
            
            
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp+xp*W+yp);
                        c_var_p_vec = _mm256_loadu_ps(color_var[i][xp]+yp);
                        c_var_q_vec = _mm256_loadu_ps(color_var[i][xq]+yq);
                        c_p_vec = _mm256_loadu_ps(color[i][xp]+yp);
                        c_q_vec = _mm256_loadu_ps(color[i][xq]+yq);

                        normalization_vec = _mm256_add_ps(c_var_p_vec, c_var_q_vec);
                        sqdist_vec = _mm256_sub_ps(c_p_vec, c_q_vec);
                        var_cancel_vec = _mm256_min_ps(c_var_p_vec, c_var_q_vec);

                        normalization_vec = _mm256_mul_ps(k_c_squared_vec, normalization_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_add_ps(c_var_p_vec, var_cancel_vec);

                        normalization_vec = _mm256_add_ps(EPSILON_vec, normalization_vec);

                        sqdist_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);
                        sqdist_vec = _mm256_div_ps(sqdist_vec, normalization_vec);

                        temp_vec = _mm256_add_ps(temp_vec, sqdist_vec);
                        _mm256_storeu_ps(temp+xp*W+yp, temp_vec);
                    }
                }
            }

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients+j * WH + xp * W + yp);
                        feat_weight = _mm256_loadu_ps(feature_weights+ xp * W + yp);

                        sqdist_vec = _mm256_sub_ps(features_p_vec, features_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_min_ps(features_var_q_vec, features_var_p_vec);
                        var_cancel_vec = _mm256_add_ps(features_var_p_vec, var_cancel_vec);
                        dist_var_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);                        

                        var_max_vec = _mm256_max_ps(features_var_p_vec, grad_vec);
                        normalization_vec = _mm256_max_ps(tau_vec, var_max_vec);
                        normalization_vec = _mm256_mul_ps(k_f_squared_vec, normalization_vec);

                        normalization_vec = _mm256_div_ps(dist_var_vec, normalization_vec);
                        feat_weight = _mm256_min_ps(feat_weight, normalization_vec);

                        _mm256_storeu_ps(feature_weights + xp * W + yp, feat_weight);
                    } 
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + p.f; yp < H - R - p.f; yp+=64) {

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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F=1 => 2*f+1 = 3
                    __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+0*8));
                    __m256 temp2_vec_1 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+1*8));
                    __m256 temp2_vec_2 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+2*8));
                    __m256 temp2_vec_3 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+0*8));
                    temp2_vec_1 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+1*8));
                    temp2_vec_2 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+2*8));
                    temp2_vec_3 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_vec_3);                    
                    // end of unrolled summation 

                    // Calculate Final Weights
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, neigh_inv_vec);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, neigh_inv_vec);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, neigh_inv_vec);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, neigh_inv_vec);

                    __m256 feat_weight_0 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

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
                        __m256 input_vec_0 = _mm256_loadu_ps(color[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(color[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(color[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(color[i][xq] + (yq + 3 * 8));

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

                for(int yp = final_yp; yp < H - R - F; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    // Unrolled Summation => Fixed for F=1 => 2*f+1 = 3
                    sum_0 += temp2[(xp-1) * W + yp+0];
                    sum_1 += temp2[(xp-1) * W + yp+1];    
                    sum_2 += temp2[(xp-1) * W + yp+2];    
                    sum_3 += temp2[(xp-1) * W + yp+3];

                    sum_0 += temp2[(xp) * W + yp+0];
                    sum_1 += temp2[(xp) * W + yp+1];
                    sum_2 += temp2[(xp) * W + yp+2];
                    sum_3 += temp2[(xp) * W + yp+3];

                    sum_0 += temp2[(xp+1) * W + yp+0];
                    sum_1 += temp2[(xp+1) * W + yp+1];
                    sum_2 += temp2[(xp+1) * W + yp+2];
                    sum_3 += temp2[(xp+1) * W + yp+3];


                    // Compute final weight
                    scalar weight_0 = exp(fmin((sum_0 * NEIGH_INV), feature_weights[xp * W + yp + 0]));
                    scalar weight_1 = exp(fmin((sum_1 * NEIGH_INV), feature_weights[xp * W + yp + 1]));
                    scalar weight_2 = exp(fmin((sum_2 * NEIGH_INV), feature_weights[xp * W + yp + 2]));
                    scalar weight_3 = exp(fmin((sum_3 * NEIGH_INV), feature_weights[xp * W + yp + 3]));

                    weight_sum[xp * W + yp + 0] += weight_0;
                    weight_sum[xp * W + yp + 1] += weight_1;
                    weight_sum[xp * W + yp + 2] += weight_2;
                    weight_sum[xp * W + yp + 3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp+0] += weight_0 * color[i][xq][yq+0];
                        output[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
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

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F; xp++){
            for (int yp = R + F ; yp < H - R - F; yp++){
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


void candidate_filtering_SECOND_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    scalar K_C_SQUARED = p.kc * p.kc;
    scalar K_F_SQUARED = p.kf * p.kf;
    scalar tau = p.tau;
    int R = p.r;
    int F = p.f;
    const __m256 k_c_squared_vec = _mm256_set1_ps(K_C_SQUARED);
    const __m256 k_f_squared_vec = _mm256_set1_ps(K_F_SQUARED);
    const __m256 tau_vec = _mm256_set1_ps(tau);

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));
    const __m256 neigh_inv_vec = _mm256_set1_ps(NEIGH_INV);


    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar sqdist, var_cancel, dist_var, var_term, var_max, normalization, df;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;
    scalar final_yp = H - 32 + R + F;

    // ------------------------
    // MEMORY ALLOCATION
    // ------------------------

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
    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1, tmp_2, grad_vec;
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+F; x < W - R - F; ++x) {
            for(int y =  R+F; y < H -  R - F; y+=8) {
                
                // Loading
                features_vec  = _mm256_loadu_ps(features[i][x] + y);
                diffL_sqr_vec = _mm256_loadu_ps(features[i][x-1] + y);
                diffR_sqr_vec =_mm256_loadu_ps(features[i][x+1]+ y);
                diffU_sqr_vec =_mm256_loadu_ps(features[i][x] + y-1);
                diffD_sqr_vec =_mm256_loadu_ps(features[i][x] + y+1);

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

                _mm256_storeu_ps(gradients+i * WH + x * W + y, tmp_1);

            } 
        }
    }

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){

            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));

            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight;
            
            
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp+xp*W+yp);
                        c_var_p_vec = _mm256_loadu_ps(color_var[i][xp]+yp);
                        c_var_q_vec = _mm256_loadu_ps(color_var[i][xq]+yq);
                        c_p_vec = _mm256_loadu_ps(color[i][xp]+yp);
                        c_q_vec = _mm256_loadu_ps(color[i][xq]+yq);

                        normalization_vec = _mm256_add_ps(c_var_p_vec, c_var_q_vec);
                        sqdist_vec = _mm256_sub_ps(c_p_vec, c_q_vec);
                        var_cancel_vec = _mm256_min_ps(c_var_p_vec, c_var_q_vec);

                        normalization_vec = _mm256_mul_ps(k_c_squared_vec, normalization_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_add_ps(c_var_p_vec, var_cancel_vec);

                        normalization_vec = _mm256_add_ps(EPSILON_vec, normalization_vec);

                        sqdist_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);
                        sqdist_vec = _mm256_div_ps(sqdist_vec, normalization_vec);

                        temp_vec = _mm256_add_ps(temp_vec, sqdist_vec);
                        _mm256_storeu_ps(temp+xp*W+yp, temp_vec);
                    }
                }
            }

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients+j * WH + xp * W + yp);
                        feat_weight = _mm256_loadu_ps(feature_weights+ xp * W + yp);

                        sqdist_vec = _mm256_sub_ps(features_p_vec, features_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_min_ps(features_var_q_vec, features_var_p_vec);
                        var_cancel_vec = _mm256_add_ps(features_var_p_vec, var_cancel_vec);
                        dist_var_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);                        

                        var_max_vec = _mm256_max_ps(features_var_p_vec, grad_vec);
                        normalization_vec = _mm256_max_ps(tau_vec, var_max_vec);
                        normalization_vec = _mm256_mul_ps(k_f_squared_vec, normalization_vec);

                        normalization_vec = _mm256_div_ps(dist_var_vec, normalization_vec);
                        feat_weight = _mm256_min_ps(feat_weight, normalization_vec);

                        _mm256_storeu_ps(feature_weights + xp * W + yp, feat_weight);
                    } 
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + p.f; yp < H - R - p.f; yp+=64) {

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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    __m256 temp2_vec_0 = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp-3) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp-2) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp-1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp+1) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp+2) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3); 

                    temp2_vec_0 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2+(xp+3) * W + (yp+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    
                    // end of unrolled summation 

                    // Calculate Final Weights
                    __m256 weight_vec_0 = _mm256_mul_ps(sum_0_vec, neigh_inv_vec);
                    __m256 weight_vec_1 = _mm256_mul_ps(sum_1_vec, neigh_inv_vec);
                    __m256 weight_vec_2 = _mm256_mul_ps(sum_2_vec, neigh_inv_vec);
                    __m256 weight_vec_3 = _mm256_mul_ps(sum_3_vec, neigh_inv_vec);

                    __m256 feat_weight_0 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

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
                        __m256 input_vec_0 = _mm256_loadu_ps(color[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(color[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(color[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(color[i][xq] + (yq + 3 * 8));

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

                for(int yp = final_yp; yp < H - R - F; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    // Unrolled Summation => Fixed for F=1 => 2*f+1 = 3
                    sum_0 += temp2[(xp-3) * W + yp];
                    sum_1 += temp2[(xp-3) * W + yp+1];
                    sum_2 += temp2[(xp-3) * W + yp+2];
                    sum_3 += temp2[(xp-3) * W + yp+3];

                    sum_0 += temp2[(xp-2) * W + yp];
                    sum_1 += temp2[(xp-2) * W + yp+1];
                    sum_2 += temp2[(xp-2) * W + yp+2];
                    sum_3 += temp2[(xp-2) * W + yp+3];

                    sum_0 += temp2[(xp-1) * W + yp];
                    sum_1 += temp2[(xp-1) * W + yp+1];
                    sum_2 += temp2[(xp-1) * W + yp+2];
                    sum_3 += temp2[(xp-1) * W + yp+3];

                    sum_0 += temp2[(xp) * W + yp];
                    sum_1 += temp2[(xp) * W + yp+1];
                    sum_2 += temp2[(xp) * W + yp+2];
                    sum_3 += temp2[(xp) * W + yp+3];

                    sum_0 += temp2[(xp+1) * W + yp];
                    sum_1 += temp2[(xp+1) * W + yp+1];
                    sum_2 += temp2[(xp+1) * W + yp+2];
                    sum_3 += temp2[(xp+1) * W + yp+3];
                    
                    sum_0 += temp2[(xp+2) * W + yp];
                    sum_1 += temp2[(xp+2) * W + yp+1];
                    sum_2 += temp2[(xp+2) * W + yp+2];
                    sum_3 += temp2[(xp+2) * W + yp+3];

                    sum_0 += temp2[(xp+3) * W + yp];
                    sum_1 += temp2[(xp+3) * W + yp+1];
                    sum_2 += temp2[(xp+3) * W + yp+2];
                    sum_3 += temp2[(xp+3) * W + yp+3];


                    // Compute final weight
                    scalar weight_0 = exp(fmin((sum_0 * NEIGH_INV), feature_weights[xp * W + yp + 0]));
                    scalar weight_1 = exp(fmin((sum_1 * NEIGH_INV), feature_weights[xp * W + yp + 1]));
                    scalar weight_2 = exp(fmin((sum_2 * NEIGH_INV), feature_weights[xp * W + yp + 2]));
                    scalar weight_3 = exp(fmin((sum_3 * NEIGH_INV), feature_weights[xp * W + yp + 3]));

                    weight_sum[xp * W + yp + 0] += weight_0;
                    weight_sum[xp * W + yp + 1] += weight_1;
                    weight_sum[xp * W + yp + 2] += weight_2;
                    weight_sum[xp * W + yp + 3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp+0] += weight_0 * color[i][xq][yq+0];
                        output[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
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

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F; xp++){
            for (int yp = R + F ; yp < H - R - F; yp++){
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



void candidate_filtering_THIRD_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);

    scalar K_C_SQUARED = p.kc * p.kc;
    scalar K_F_SQUARED = p.kf * p.kf;
    scalar tau = p.tau;
    int R = p.r;
    int F = p.f;
    const __m256 k_c_squared_vec = _mm256_set1_ps(K_C_SQUARED);
    const __m256 k_f_squared_vec = _mm256_set1_ps(K_F_SQUARED);
    const __m256 tau_vec = _mm256_set1_ps(tau);

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));
    const __m256 neigh_inv_vec = _mm256_set1_ps(NEIGH_INV);


    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar sqdist, var_cancel, dist_var, var_term, var_max, normalization, df;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;
    scalar final_yp = H - 32 + R + F;

    // ------------------------
    // MEMORY ALLOCATION
    // ------------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    weight_sum = (scalar*) calloc(W * H, sizeof(scalar));

    // Init feature weights channel
    scalar* feature_weights;
    feature_weights = (scalar*) malloc(W * H * sizeof(scalar));

    // Compute gradients
    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1, tmp_2, grad_vec;
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+F; x < W - R - F; ++x) {
            for(int y =  R+F; y < H -  R - F; y+=8) {
                
                // Loading
                features_vec  = _mm256_loadu_ps(features[i][x] + y);
                diffL_sqr_vec = _mm256_loadu_ps(features[i][x-1] + y);
                diffR_sqr_vec =_mm256_loadu_ps(features[i][x+1]+ y);
                diffU_sqr_vec =_mm256_loadu_ps(features[i][x] + y-1);
                diffD_sqr_vec =_mm256_loadu_ps(features[i][x] + y+1);

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

                _mm256_storeu_ps(gradients+i * WH + x * W + y, tmp_1);

            } 
        }
    }

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){

            // Compute Color Weight for all pixels with fixed r

            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight;
            

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients+j * WH + xp * W + yp);
                        feat_weight = _mm256_loadu_ps(feature_weights+ xp * W + yp);

                        sqdist_vec = _mm256_sub_ps(features_p_vec, features_q_vec);
                        sqdist_vec = _mm256_mul_ps(sqdist_vec, sqdist_vec);
                        var_cancel_vec = _mm256_min_ps(features_var_q_vec, features_var_p_vec);
                        var_cancel_vec = _mm256_add_ps(features_var_p_vec, var_cancel_vec);
                        dist_var_vec = _mm256_sub_ps(var_cancel_vec, sqdist_vec);                        

                        var_max_vec = _mm256_max_ps(features_var_p_vec, grad_vec);
                        normalization_vec = _mm256_max_ps(tau_vec, var_max_vec);
                        normalization_vec = _mm256_mul_ps(k_f_squared_vec, normalization_vec);

                        normalization_vec = _mm256_div_ps(dist_var_vec, normalization_vec);
                        feat_weight = _mm256_min_ps(feat_weight, normalization_vec);

                        _mm256_storeu_ps(feature_weights + xp * W + yp, feat_weight);
                    } 
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - 32; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 weight_vec_0 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 0 * 8));
                    __m256 weight_vec_1 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 1 * 8));
                    __m256 weight_vec_2 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 2 * 8));
                    __m256 weight_vec_3 = _mm256_loadu_ps(feature_weights+ xp * W + (yp + 3 * 8));

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
                        __m256 input_vec_0 = _mm256_loadu_ps(color[i][xq] + (yq + 0 * 8));
                        __m256 output_vec_1 = _mm256_loadu_ps(output[i][xp] + (yp + 1 * 8));
                        __m256 input_vec_1 = _mm256_loadu_ps(color[i][xq] + (yq + 1 * 8));
                        __m256 output_vec_2 = _mm256_loadu_ps(output[i][xp] + (yp + 2 * 8));
                        __m256 input_vec_2 = _mm256_loadu_ps(color[i][xq] + (yq + 2 * 8));
                        __m256 output_vec_3 = _mm256_loadu_ps(output[i][xp] + (yp + 3 * 8));
                        __m256 input_vec_3 = _mm256_loadu_ps(color[i][xq] + (yq + 3 * 8));

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
                for(int yp = final_yp; yp < H - R - F; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar weight_0 = exp(feature_weights[xp * W + yp]);
                    scalar weight_1 = exp(feature_weights[xp * W + yp+1]);
                    scalar weight_2 = exp(feature_weights[xp * W + yp+2]);
                    scalar weight_3 = exp(feature_weights[xp * W + yp+3]);

                    weight_sum[xp * W + yp] += weight_0;
                    weight_sum[xp * W + yp+1] += weight_1;
                    weight_sum[xp * W + yp+2] += weight_2;
                    weight_sum[xp * W + yp+3] += weight_3;

                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
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

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F; xp++){
            for (int yp = R + F ; yp < H - R - F; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
            }
        }
    }

    // Free memory
    free(weight_sum);
    free(feature_weights);
    free(gradients);

}


void candidate_filtering_all_BLK(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int X0, int Y0, int W, int H){

    int WH = W*H;

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


    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(F_R, fmax(F_G, F_B));
    int f_min = fmin(F_R, fmin(F_G, F_B));
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
    const int tempW = W + 2*f_max + 1;
    const int tempH = H + 2*f_max + 1;
    temp = (scalar*) malloc(tempW * tempH * sizeof(scalar));
    temp2_r = (scalar*) malloc(tempW * tempH * sizeof(scalar));
    temp2_g = (scalar*) malloc(tempW * tempH * sizeof(scalar));

    // Allocate feature weights buffer
    scalar* features_weights_r;
    scalar* features_weights_b;
    features_weights_r = (scalar*) malloc(W * H * sizeof(scalar));
    features_weights_b = (scalar*) malloc(W * H * sizeof(scalar));

    // Precompute size of neighbourhood
    scalar NEIGH_R_INV = 1. / (3*(2*F_R+1)*(2*F_R+1));
    scalar NEIGH_G_INV = 1. / (3*(2*F_G+1)*(2*F_G+1));
    int final_yp_r = Y0+H - 32;
    int final_yp_g = Y0+H - 32;

    const __m256 neigh_inv_r_vec = _mm256_set1_ps(NEIGH_R_INV);
    const __m256 neigh_inv_min_r_vec = _mm256_set1_ps(-NEIGH_R_INV);
    const __m256 neigh_inv_g_vec = _mm256_set1_ps(NEIGH_G_INV);
    const __m256 neigh_inv_min_g_vec = _mm256_set1_ps(-NEIGH_G_INV);   
    
    // Compute gradients
    __m256 features_vec, diffL_sqr_vec, diffR_sqr_vec, diffU_sqr_vec, diffD_sqr_vec, tmp_1, tmp_2, grad_vec;
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  X0; x < X0+W; ++x) {
            for(int y =  Y0; y < Y0+H; y+=8) {
                
                // Loading
                features_vec  = _mm256_loadu_ps(features[i][x] + y);
                diffL_sqr_vec = _mm256_loadu_ps(features[i][x-1] + y);
                diffR_sqr_vec =_mm256_loadu_ps(features[i][x+1]+ y);
                diffU_sqr_vec =_mm256_loadu_ps(features[i][x] + y-1);
                diffD_sqr_vec =_mm256_loadu_ps(features[i][x] + y+1);

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

                _mm256_storeu_ps(gradients+i * WH + (x-X0) * W + (y-Y0), tmp_1);

            } 
        }
    }

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // #######################################################################################
            // WEIGHT COMPUTATION
            // #######################################################################################

            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, tempW*tempH*sizeof(scalar));

            __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, normalization_vec, temp_vec;
            __m256 features_p_vec, features_q_vec, features_var_p_vec, features_var_q_vec;
            __m256 var_cancel_vec, var_max_vec, gradient_vec, normalization_r_vec, normalization_b_vec; 
            __m256 dist_var_vec, tmp_1, tmp_2, feat_weight_r, feat_weight_b;
            
            
            for (int i=0; i<3; i++){  
                for(int xp = X0-f_max; xp < X0+W+f_max; ++xp) {
                    int yp = Y0-f_max;
                    for(yp; yp < Y0+H+f_max-8; yp+=8) {

                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        temp_vec = _mm256_loadu_ps(temp+(xp-X0+f_max)*tempW+(yp-Y0+f_max));
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
                        _mm256_storeu_ps(temp+(xp-X0+f_max)*tempW+(yp-Y0+f_max), temp_vec);
                    }
                    if(yp > Y0+H+f_max-8)
                        yp -= 8;
                    for(yp; yp < Y0+H+f_max; ++yp) {
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        scalar temp_sca = temp[(xp-X0+f_max)*tempW+(yp-Y0+f_max)];
                        scalar c_var_p = color_var[i][xp][yp];
                        scalar c_var_q = color_var[i][xq][yq];
                        scalar c_p = color[i][xp][yp];
                        scalar c_q = color[i][xq][yq];

                        scalar normalization = c_var_p + c_var_q;
                        scalar sqdist = c_p - c_q;
                        scalar var_cancel = fmin(c_var_p, c_var_q);

                        normalization = k_c_squared_r * normalization;
                        sqdist *= sqdist;
                        var_cancel = c_var_p + var_cancel;

                        normalization += EPSILON;

                        sqdist = var_cancel - sqdist;
                        sqdist /= normalization;

                        temp_sca += sqdist;
                        temp[(xp-X0+f_max)*tempW+(yp-Y0+f_max)] = temp_sca;
                    }
                }
            }
           
            // Precompute feature weights

            
            memset(features_weights_r, 0, W*H*sizeof(scalar));
            memset(features_weights_b, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = X0; xp < X0+W; ++xp) {
                    int yp = Y0;
                    for(yp; yp < Y0+H-8; yp+=8) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        features_p_vec = _mm256_loadu_ps(features[j][xp] + yp);
                        features_q_vec = _mm256_loadu_ps(features[j][xq] + yq);
                        features_var_p_vec = _mm256_loadu_ps(features_var[j][xp] + yp);
                        features_var_q_vec = _mm256_loadu_ps(features_var[j][xq] + yq);
                        grad_vec = _mm256_loadu_ps(gradients+j * WH + (xp-X0) * W + (yp-Y0));
                        feat_weight_r = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + (yp-Y0));
                        feat_weight_b = _mm256_loadu_ps(features_weights_b+ (xp-X0) * W + (yp-Y0));

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

                        _mm256_storeu_ps(features_weights_r + (xp-X0) * W + (yp-Y0), feat_weight_r);
                        _mm256_storeu_ps(features_weights_b + (xp-X0) * W + (yp-Y0), feat_weight_b);
                    } 
                    if(yp > Y0+H-8)
                        yp -= 8;
                    for(yp; yp < Y0+H; yp++) {
                        
                        int xq = xp + r_x;
                        int yq = yp + r_y;

                        scalar features_p = features[j][xp][yp];
                        scalar features_q = features[j][xq][yq];
                        scalar features_var_p = features_var[j][xp][yp];
                        scalar features_var_q = features_var[j][xq][yq];
                        scalar grad = *(gradients+j * WH + (xp-X0) * W + (yp-Y0));
                        scalar feat_weight_r = *(features_weights_r+ (xp-X0) * W + (yp-Y0));
                        scalar feat_weight_b = *(features_weights_b+ (xp-X0) * W + (yp-Y0));

                        scalar sqdist = (features_p - features_q);
                        sqdist = (sqdist * sqdist);
                        scalar var_cancel = fmin(features_var_q, features_var_p);
                        var_cancel = (features_var_p + var_cancel);
                        scalar dist_var = (var_cancel - sqdist);                        

                        scalar var_max = fmax(features_var_p, grad);
                        scalar normalization_r = fmax(tau_r, var_max);
                        scalar normalization_b = fmax(tau_b, var_max);
                        normalization_r = (k_f_squared_r * normalization_r);
                        normalization_b = (k_f_squared_b * normalization_b);

                        normalization_r = (dist_var / normalization_r);
                        normalization_b = (dist_var / normalization_b);
                        feat_weight_r = fmin(feat_weight_r, normalization_r);
                        feat_weight_b = fmin(feat_weight_b, normalization_b);

                        *(features_weights_r + (xp-X0) * W + (yp-Y0)) = feat_weight_r;
                        *(features_weights_b + (xp-X0) * W + (yp-Y0)) = feat_weight_b;
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
            for(int xp = X0-F_R; xp < X0+W+F_R; ++xp) {
                for(int yp = Y0; yp < Y0+H; yp+=32) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-F_R; k<=F_R; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + (yp-Y0+f_max)+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + ((yp-Y0+f_max)+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + ((yp-Y0+f_max)+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + ((yp-Y0+f_max)+3*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                    }

                    _mm256_storeu_ps(temp2_r+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_r+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_r+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_r+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 3 * 8), sum_3_vec);
                }
                // assert(H % 64 == 0);
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = X0; xp < X0+W; ++xp) {
                for(int yp = Y0; yp < Y0+H; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_R=1 => 2*f+1 = 3
                    __m256 temp2_vec_0 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_r+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+3*8));
                    
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

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_r+ (xp-X0) * W + ( (yp-Y0) + 3 * 8), weight_sum_vec_3);
                    
                    
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
                // assert(H % 32 == 0);
                /*
                for(int yp = final_yp_r; yp < Y0+H; yp++) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;

                    // Unrolled Summation => Fixed for F_R=1 => 2*f+1 = 3
                    sum_0 += temp2_r[((xp-X0+f_max)-1) * tempW + (yp-Y0+f_max)+0];

                    sum_0 += temp2_r[((xp-X0+f_max)) * tempW + (yp-Y0+f_max)+0];

                    sum_0 += temp2_r[((xp-X0+f_max)+1) * tempW + (yp-Y0+f_max)+0];


                    // Compute final weight
                    scalar weight_0 = exp(fmin((sum_0 * NEIGH_R_INV), features_weights_r[(xp-X0) * W + (yp-Y0) + 0]));

                    weight_sum_r[(xp-X0) * W + (yp-Y0) + 0] += weight_0;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp+0] += weight_0 * color[i][xq][yq+0];
                    }
                    
                }*/
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            for(int xp = X0-F_G; xp < X0+W+F_G; ++xp) {
                for(int yp = Y0; yp < Y0+H; yp+=32) {

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                    for (int k=-F_G; k<=F_G; k++){

                        __m256 temp_vec_0 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + (yp-Y0+f_max)+k);
                        __m256 temp_vec_1 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + ((yp-Y0+f_max)+1*8)+k);
                        __m256 temp_vec_2 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + ((yp-Y0+f_max)+2*8)+k);
                        __m256 temp_vec_3 = _mm256_loadu_ps(temp+(xp-X0+f_max) * tempW + ((yp-Y0+f_max)+3*8)+k);

                        sum_0_vec = _mm256_add_ps(sum_0_vec, temp_vec_0);
                        sum_1_vec = _mm256_add_ps(sum_1_vec, temp_vec_1);
                        sum_2_vec = _mm256_add_ps(sum_2_vec, temp_vec_2);
                        sum_3_vec = _mm256_add_ps(sum_3_vec, temp_vec_3);
                    }

                    _mm256_storeu_ps(temp2_g+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 0 * 8), sum_0_vec);
                    _mm256_storeu_ps(temp2_g+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 1 * 8), sum_1_vec);
                    _mm256_storeu_ps(temp2_g+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 2 * 8), sum_2_vec);
                    _mm256_storeu_ps(temp2_g+ (xp-X0+f_max) * tempW + ((yp-Y0+f_max) + 3 * 8), sum_3_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = X0; xp < X0+W; ++xp) {
                for(int yp = Y0; yp < Y0+H; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_0_vec = _mm256_setzero_ps();
                    __m256 sum_1_vec = _mm256_setzero_ps();
                    __m256 sum_2_vec = _mm256_setzero_ps();
                    __m256 sum_3_vec = _mm256_setzero_ps();

                     // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    __m256 temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-3) * tempW + ((yp-Y0+f_max)+0*8));
                    __m256 temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-3) * tempW + ((yp-Y0+f_max)+1*8));
                    __m256 temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-3) * tempW + ((yp-Y0+f_max)+2*8));
                    __m256 temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-3) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-2) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-2) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-2) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-2) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)-1) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);                    

                    temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+1) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3);

                    temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+2) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+2) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+2) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0+f_max)+2) * tempW + ((yp-Y0+f_max)+3*8));
                    
                    sum_0_vec = _mm256_add_ps(sum_0_vec, temp2_vec_0);
                    sum_1_vec = _mm256_add_ps(sum_1_vec, temp2_r_vec_1);
                    sum_2_vec = _mm256_add_ps(sum_2_vec, temp2_r_vec_2);
                    sum_3_vec = _mm256_add_ps(sum_3_vec, temp2_r_vec_3); 

                    temp2_vec_0 = _mm256_loadu_ps(temp2_g+((xp-X0)+3) * tempW + ((yp-Y0+f_max)+0*8));
                    temp2_r_vec_1 = _mm256_loadu_ps(temp2_g+((xp-X0)+3) * tempW + ((yp-Y0+f_max)+1*8));
                    temp2_r_vec_2 = _mm256_loadu_ps(temp2_g+((xp-X0)+3) * tempW + ((yp-Y0+f_max)+2*8));
                    temp2_r_vec_3 = _mm256_loadu_ps(temp2_g+((xp-X0)+3) * tempW + ((yp-Y0+f_max)+3*8));
                    
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

                    __m256 feat_weight_0 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 0 * 8));
                    __m256 feat_weight_1 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 1 * 8));
                    __m256 feat_weight_2 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 2 * 8));
                    __m256 feat_weight_3 = _mm256_loadu_ps(features_weights_r+ (xp-X0) * W + ((yp-Y0) + 3 * 8));

                    weight_vec_0 = _mm256_min_ps(feat_weight_0, weight_vec_0);
                    weight_vec_1 = _mm256_min_ps(feat_weight_1, weight_vec_1);
                    weight_vec_2 = _mm256_min_ps(feat_weight_2, weight_vec_2);
                    weight_vec_3 = _mm256_min_ps(feat_weight_3, weight_vec_3);

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_g+ (xp-X0) * W + ( (yp-Y0) + 3 * 8), weight_sum_vec_3);
                    
                    
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
                /*
                for(int yp = final_yp_g; yp < Y0+H; yp++) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;

                    // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    sum_0 += temp2_g[((xp-X0+f_max)-3) * tempW + (yp-Y0+f_max)];

                    sum_0 += temp2_g[((xp-X0+f_max)-2) * tempW + (yp-Y0+f_max)];

                    sum_0 += temp2_g[((xp-X0+f_max)-1) * tempW + (yp-Y0+f_max)];

                    sum_0 += temp2_g[((xp-X0+f_max)) * tempW + (yp-Y0+f_max)];

                    sum_0 += temp2_g[((xp-X0+f_max)+1) * tempW + (yp-Y0+f_max)];
                    
                    sum_0 += temp2_g[((xp-X0+f_max)+2) * tempW + (yp-Y0+f_max)];

                    sum_0 += temp2_g[((xp-X0+f_max)+3) * tempW + (yp-Y0+f_max)];


                    // Compute final weight
                    scalar weight_0 = exp(fmin((sum_0 * NEIGH_G_INV), features_weights_r[(xp-X0) * W + (yp-Y0) + 0]));

                    weight_sum_g[(xp-X0) * W + (yp-Y0) + 0] += weight_0;
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp+0] += weight_0 * color[i][xq][yq+0];
                    }
                    
                }*/
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = X0; xp < X0+W; ++xp) {
                for(int yp = Y0; yp < Y0+H; yp+=32) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 weight_vec_0 = _mm256_loadu_ps(features_weights_b+ (xp-X0) * W + ((yp-Y0) + 0 * 8));
                    __m256 weight_vec_1 = _mm256_loadu_ps(features_weights_b+ (xp-X0) * W + ((yp-Y0) + 1 * 8));
                    __m256 weight_vec_2 = _mm256_loadu_ps(features_weights_b+ (xp-X0) * W + ((yp-Y0) + 2 * 8));
                    __m256 weight_vec_3 = _mm256_loadu_ps(features_weights_b+ (xp-X0) * W + ((yp-Y0) + 3 * 8));

                    weight_vec_0 = exp256_ps(weight_vec_0);
                    weight_vec_1 = exp256_ps(weight_vec_1);
                    weight_vec_2 = exp256_ps(weight_vec_2);
                    weight_vec_3 = exp256_ps(weight_vec_3);

                    __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 0 * 8));
                    __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 1 * 8));
                    __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 2 * 8));
                    __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 3 * 8));

                    weight_sum_vec_0 = _mm256_add_ps(weight_sum_vec_0, weight_vec_0);
                    weight_sum_vec_1 = _mm256_add_ps(weight_sum_vec_1, weight_vec_1);
                    weight_sum_vec_2 = _mm256_add_ps(weight_sum_vec_2, weight_vec_2);
                    weight_sum_vec_3 = _mm256_add_ps(weight_sum_vec_3, weight_vec_3);

                    _mm256_storeu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 0 * 8), weight_sum_vec_0);
                    _mm256_storeu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 1 * 8), weight_sum_vec_1);
                    _mm256_storeu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 2 * 8), weight_sum_vec_2);
                    _mm256_storeu_ps(weight_sum_b+ (xp-X0) * W + ( (yp-Y0) + 3 * 8), weight_sum_vec_3);
                    
                    

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
                /*
                for(int yp = final_yp_r; yp < Y0+H; yp++) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar weight_0 = exp(features_weights_b[(xp-X0) * W + (yp-Y0)]);

                    weight_sum_b[(xp-X0) * W + (yp-Y0)] += weight_0;

                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp] += weight_0 * color[i][xq][yq];
                    }
                }   */         
            }
            
        }
    }

    // Final Weight Normalization R
    for(int xp = X0; xp < X0+W; ++xp) {
        for(int yp = Y0 ; yp < Y0+H; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_r + (xp-X0) * W + ((yp-Y0) + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_r + (xp-X0) * W + ((yp-Y0) + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_r + (xp-X0) * W + ((yp-Y0) + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_r + (xp-X0) * W + ((yp-Y0) + 3 * 8));


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
        /*
        for(int yp = final_yp_r; yp < Y0+H; yp++) {     
            
            scalar w_0 = weight_sum_r[(xp-X0) * W + (yp-Y0)];

            for (int i=0; i<3; i++){
                output_r[i][xp][yp] /= w_0;
            }
        }*/
    }    


    // Final Weight Normalization G
    for(int xp = X0; xp < X0+W; ++xp) {
        for(int yp = Y0; yp < Y0+H; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_g + (xp-X0) * W + ((yp-Y0) + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_g + (xp-X0) * W + ((yp-Y0) + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_g + (xp-X0) * W + ((yp-Y0) + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_g + (xp-X0) * W + ((yp-Y0) + 3 * 8));


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
        /*
        for(int yp = final_yp_g; yp < Y0+H; yp++) {     
            
            scalar w_0 = weight_sum_g[(xp-X0) * W + (yp-Y0)];

            for (int i=0; i<3; i++){
                output_g[i][xp][yp] /= w_0;
            }
        }*/
    }

    // Final Weight Normalization B
    // Final Weight Normalization
    for(int xp = X0; xp < X0+W; ++xp) {
        for(int yp = Y0 ; yp < Y0+H; yp+=32) {     

            __m256 weight_sum_vec_0 = _mm256_loadu_ps(weight_sum_b + (xp-X0) * W + ((yp-Y0) + 0 * 8));
            __m256 weight_sum_vec_1 = _mm256_loadu_ps(weight_sum_b + (xp-X0) * W + ((yp-Y0) + 1 * 8));
            __m256 weight_sum_vec_2 = _mm256_loadu_ps(weight_sum_b + (xp-X0) * W + ((yp-Y0) + 2 * 8));
            __m256 weight_sum_vec_3 = _mm256_loadu_ps(weight_sum_b + (xp-X0) * W + ((yp-Y0) + 3 * 8));


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
        /*
        for(int yp = final_yp_r; yp < Y0+H; yp++) {     
            
            scalar w_0 = weight_sum_b[(xp-X0) * W + (yp-Y0)];

            for (int i=0; i<3; i++){
                output_b[i][xp][yp] /= w_0;
            }
        }*/
    }

    // Free memory
    free(weight_sum_r);
    free(weight_sum_g);
    free(weight_sum_b);
    free(temp);
    free(temp2_r);
    free(temp2_g);
    free(features_weights_r);
    free(features_weights_b);
    //free(norm_r);
    //free(norm_b);
    free(gradients);

}

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
                c_vec = _mm256_loadu_ps(c+(i * WH + x * W + y));
                d_r_vec = _mm256_loadu_ps(cand_r+(i * WH + x * W + y));
                d_g_vec = _mm256_loadu_ps(cand_g+(i * WH + x * W + y));
                d_b_vec = _mm256_loadu_ps(cand_b+(i * WH + x * W + y));
                v_vec = _mm256_loadu_ps(c_var+(i * WH + x * W + y));

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
            _mm256_storeu_ps(sure+(0 * WH + x * W + y), sure_r_vec);
            _mm256_storeu_ps(sure+(1 * WH + x * W + y), sure_g_vec);
            _mm256_storeu_ps(sure+(2 * WH + x * W + y), sure_b_vec);
        }
    }
}


void filtering_basic_vec(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int W, int H){
    
    int WH = W*H;
    const __m256 EPSILON_vec = _mm256_set1_ps(EPSILON);


    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    const __m256 k_c_squared_vec = _mm256_set1_ps(k_c_squared);

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
    __m256 neigh_vec = _mm256_set1_ps(neigh);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r; yp < H - p.r; yp+=8) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    __m256 distance_vec = _mm256_setzero_ps();
                    __m256 c_p_vec, c_q_vec, sqdist_vec, c_var_p_vec, c_var_q_vec, var_cancel_vec, normalization_vec;
                    for (int i=0; i<3; i++){
                        c_var_p_vec = _mm256_loadu_ps(c_var+i * WH + xp * W + yp);
                        c_var_q_vec = _mm256_loadu_ps(c_var+i * WH + xq * W + yq);
                        c_p_vec = _mm256_loadu_ps(c+i * WH + xp * W + yp);
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
                    _mm256_storeu_ps(temp+xp*W+yp, distance_vec);

                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < W - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=8) {

                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec;
                    for (int k=-p.f; k<=p.f; k++){
                        tmp_vec = _mm256_loadu_ps(temp+xp * W + yp + k);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                    }
                    _mm256_storeu_ps(temp2+xp * W + yp, sum_vec);
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; yp+=1) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    __m256 sum_vec = _mm256_setzero_ps();
                    __m256 tmp_vec, weight_vec;
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        tmp_vec = _mm256_loadu_ps(temp2+(xp+k) * W + yp);
                        sum_vec = _mm256_add_ps(sum_vec, tmp_vec);
                        sum += temp2[(xp+k) * W + yp];
                    }
                    weight_vec = _mm256_div_ps(sum_vec, neigh_vec);
                    weight_vec = _mm256_max_ps(weight_vec, _mm256_setzero_ps());
                    weight_vec = _mm256_mul_ps(weight_vec, _mm256_set1_ps(-1.0));
                    weight_vec = exp256_ps(weight_vec);
                    // scalar weight = exp(-fmax(0.f, (sum / neigh)));
                    // weight_sum[xp * W + yp] += weight;

                    __m256 weight_sum_vec = _mm256_load_ps(weight_sum+xp * W + yp);
                    weight_sum_vec = _mm256_add_ps(weight_sum_vec, weight_vec);
                    _mm256_storeu_ps(weight_sum+xp * W + yp, weight_sum_vec);
                    
                    __m256 input_vec, output_vec;
                    for (int i=0; i<3; i++){
                        input_vec = _mm256_loadu_ps(input+i * WH + xq * W + yq);
                        output_vec = _mm256_loadu_ps(output+i * WH + xp * W + yp);
                        input_vec = _mm256_mul_ps(weight_vec, input_vec);
                        output_vec = _mm256_add_ps(input_vec, output_vec);
                        _mm256_storeu_ps(output+i * WH + xp * W + yp, output_vec);                        
                        // output[i * WH + xp * W + yp] += weight * input[i * WH + xq * W + yq];
                    }
                }
            }

        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f ; yp < H - p.r - p.f; yp+=8) {     
            __m256 weight_sum_vec, output_vec;
            // scalar w = weight_sum[xp * W + yp];
            weight_sum_vec = _mm256_loadu_ps(weight_sum+xp * W + yp);
            for (int i=0; i<3; i++){
                output_vec = _mm256_loadu_ps(output+i * WH + xp * W + yp);
                output_vec = _mm256_div_ps(output_vec, weight_sum_vec);
                // output[i * WH + xp * W + yp] /= w;
                _mm256_storeu_ps(output, output_vec);
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                output[i * WH + xp * W + H - yp - 1] = input[i * WH + xp * W + H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
             for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                output[i * WH + (W - xp - 1) * W + yp] = input[i * WH + (W - xp - 1) * W + yp];
            }
        }

    }

    free(weight_sum);
    free(temp);
    free(temp2); 
}


void feature_prefiltering_vec(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int W, int H){

    int WH = W*H;

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
                        scalar sqdist = features[i * WH + xp * W + yp] - features[i * WH + xq * W + yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[i * WH + xp * W + yp] + fmin(features_var[i * WH + xp * W + yp], features_var[i * WH + xq * W + yq]);
                        scalar normalization = EPSILON + k_c_squared*(features_var[i * WH + xp * W + yp] + features_var[i * WH + xq * W + yq]);
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
                        output[i * WH + xp * W + yp] += weight * features[i * WH + xq * W + yq];
                        output_var[i * WH + xp * W + yp] += weight * features_var[i * WH + xq * W + yq];
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
                output[i * WH + xp * W + yp] /= w;
                output_var[i * WH + xp * W + yp] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i * WH + xp * W + yp] = features[i * WH + xp * W + yp];
                output[i * WH + xp * W + H - yp - 1] = features[i * WH + xp * W + H - yp - 1];
                output_var[i * WH + xp * W + yp] = features_var[i * WH + xp * W + yp];
                output_var[i * WH + xp * W + H - yp - 1] = features_var[i * WH + xp * W + H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
             for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
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

void candidate_filtering_vec(scalar* output, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters p, int W, int H){

    int WH = W * H;

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    scalar k_f_squared = p.kf * p.kf;

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum;
    allocate_channel_zero(&weight_sum, W, H);

    // Init feature weights
    scalar* feature_weights;
    allocate_channel(&feature_weights, W, H);

    // Init temp channel
    scalar* temp;
    scalar* temp2;
    allocate_channel(&temp, W, H);
    allocate_channel(&temp2, W, H);

    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  p.r + p.f; x < W - p.r - p.f; ++x) {
            for(int y = p.r + p.f; y < H - p.r - p.f; ++y) {
                scalar diffL = features[i * WH + x * W + y] - features[i * WH + (x-1) * W + y];
                scalar diffR = features[i * WH + x * W + y] - features[i * WH + (x+1) * W + y];
                scalar diffU = features[i * WH + x * W + y] - features[i * WH + x * W + y - 1];
                scalar diffD = features[i * WH + x * W + y] - features[i * WH + x * W + y + 1];

                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

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
                        scalar sqdist = color[i * WH + xp * W + yp] - color[i * WH + xq * W + yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i * WH + xp * W + yp] + fmin(color_var[i * WH + xp * W + yp], color_var[i * WH + xq * W + yq]);
                        scalar normalization = EPSILON + k_c_squared*(color_var[i * WH + xp * W + yp] + color_var[i * WH + xq * W + yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp * W + yp] = distance;

                }
            }

            // Compute features
            for(int xp = p.r + p.f; xp < W - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < H - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;


                    // Compute feature weight
                    scalar df = 0.f;
                    for(int j=0; j<NB_FEATURES;++j){
                        scalar sqdist = features[j * WH + xp * W + yp] - features[j * WH + xq * W + yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j * WH + xp * W + yp] + fmin(features_var[j * WH + xp * W + yp], features_var[j * WH + xq * W + yq]);
                        scalar normalization = k_f_squared*fmax(p.tau, fmax(features_var[j * WH + xp * W + yp], gradients[j * WH + xp * W + yp]));
                        df = fmax(df, (sqdist - var_cancel)/normalization);
                    }
                    feature_weights[xp * W + yp] = exp(-df);
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
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh)));
                    
                    scalar weight = fmin(color_weight, feature_weights[xp * W + yp]);
                    weight_sum[xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i * WH + xp * W + yp] += weight * color[i * WH + xq * W + yq];
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
                output[i * WH + xp * W + yp] /= w;
            }
        }
    }

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r + p.f; yp++){
                output[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output[i * WH + xp * W + H - yp - 1] = color[i * WH + xp * W + H - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
            for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[i * WH + xp * W + yp] = color[i * WH + xp * W + yp];
                output[i * WH + (W - xp - 1) * W + yp] = color[i * WH + (W - xp - 1) * W + yp];
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

    // Rename Width and Height
    int WH = W * H;

    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(f_r, fmax(f_g, f_b));
    int f_min = fmin(f_r, fmin(f_g, f_b));
    int R = p[0].r;

    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    scalar *weight_sum;
    allocate_buffer_zero(&weight_sum, W, H);

    // Init temp channel
    scalar *temp;
    scalar *temp2_r;
    scalar *temp2_g;
    allocate_channel(&temp, W, H);
    allocate_channel(&temp2_r, W, H);
    allocate_channel(&temp2_g, W, H);

    // Allocate feature weights buffer
    scalar *features_weights_r;
    scalar *features_weights_b;
    allocate_channel(&features_weights_r, W, H);
    allocate_channel(&features_weights_b, W, H);

    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+f_min; x < W - R - f_min; ++x) {
            for(int y =  R+f_min; y < H-  R - f_min; ++y) {
                scalar diffL = features[i * WH + x * W + y] - features[i * WH + (x-1) * W + y];
                scalar diffR = features[i * WH + x * W + y] - features[i * WH + (x+1) * W + y];
                scalar diffU = features[i * WH + x * W + y] - features[i * WH + x * W + y - 1];
                scalar diffD = features[i * WH + x * W + y] - features[i * WH + x * W + y + 1];

                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
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

                    scalar distance_r = 0.f;

                    for (int i=0; i<3; i++){                        
                        scalar sqdist = color[i * WH + xp * W + yp] - color[i * WH + xq * W + yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i * WH + xp * W + yp] + fmin(color_var[i * WH + xp * W + yp], color_var[i * WH + xq * W + yq]);
                        scalar var_term = color_var[i * WH + xp * W + yp] + color_var[i * WH + xq * W + yq];
                        scalar normalization_r = EPSILON + k_c_squared_r*(var_term);
                        scalar dist_var = sqdist - var_cancel;
                        distance_r += (dist_var) / normalization_r;
                    }

                    temp[xp * W + yp] = distance_r;

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
                        scalar sqdist = features[j * WH + xp * W + yp] - features[j * WH + xq * W + yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j * WH + xp * W + yp] + fmin(features_var[j * WH + xp * W + yp], features_var[j * WH + xq * W + yq]);
                        scalar var_max = fmax(features_var[j * WH + xp * W + yp], gradients[j * WH + xp * W + yp]);
                        scalar normalization_r = k_f_squared_r*fmax(tau_r, var_max);
                        scalar normalization_b = k_f_squared_b*fmax(tau_b, var_max);
                        scalar dist_var = sqdist - var_cancel;
                        df_r = fmax(df_r, (dist_var)/normalization_r);
                        df_b = fmax(df_b, (dist_var)/normalization_b);
                    }

                    features_weights_r[xp * W + yp] = exp(-df_r);
                    features_weights_b[xp * W + yp] = exp(-df_b);
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
                        sum_r += temp[xp * W + yp + k];
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
                        sum += temp2_r[(xp + k) * W + yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh_r)));

                    // Compute final weight
                    scalar weight = fmin(color_weight, features_weights_r[xp * W + yp]);
                    weight_sum[0 * WH + xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_r[i * WH + xp * W + yp] += weight * color[i * WH + xq * W + yq];
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
                        sum_g += temp[xp * W + yp + k];
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
                        sum += temp2_g[(xp + k) * W + yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh_g)));
                    
                    // Compute final weight
                    scalar weight = fmin(color_weight, features_weights_r[xp * W + yp]);
                    weight_sum[1 * WH + xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_g[i * WH + xp * W + yp] += weight * color[i * WH + xq * W + yq];
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
                    scalar weight = features_weights_b[xp * W + yp];
                    weight_sum[2 * WH + xp * W + yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_b[i * WH + xp * W + yp] += weight * color[i * WH + xq * W + yq];
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < H - R - f_r; ++yp) {
        
            scalar w = weight_sum[0 * WH + xp * W + yp];
            for (int i=0; i<3; i++){
                output_r[i * WH + xp * W + yp] /= w;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < H - R - f_g; ++yp) {
        
            scalar w = weight_sum[1 * WH + xp * W + yp];
            for (int i=0; i<3; i++){
                output_g[i * WH + xp * W + yp] /= w;
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < H - R - f_b; ++yp) {
        
            scalar w = weight_sum[2 * WH + xp * W + yp];
            for (int i=0; i<3; i++){
                output_b[i * WH + xp * W + yp] /= w;
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

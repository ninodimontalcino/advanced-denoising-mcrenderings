#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all_ILP(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H){
    
    // $unroll 1
    scalar col$i, d_r$i, d_g$i, d_b$i, v$i;
    
    for (int i = 0; i < 3; i++){ 
        for (int x = 0; x < W; x++){
            for (int y = 0; y < H; y+=$n){

                col$i = c[i][x][y+$i];

                // Calculate terms
                d_r$i = cand_r[i][x][y+$i] - col$i;
                d_r$i *= d_r$i;
                d_g$i = cand_g[i][x][y+$i] - col$i;
                d_g$i *= d_g$i;
                d_b$i = cand_b[i][x][y+$i] - col$i;
                d_b$i *= d_b$i;
                v$i = c_var[i][x][y+$i];
                v$i *= v$i;
                
                // Store sure error estimate
                sure[0][x][y+$i] += d_r$i - v$i;
                sure[1][x][y+$i] += d_g$i - v$i;
                sure[2][x][y+$i] += d_b$i - v$i;
            }
        }
    }
    // $end_unroll
}


void filtering_basic_f3_ILP(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    

    scalar K_C_SQUARED = p.kc * p.kc;
    int R = p.r;
    int F = p.f;

    int xq, yq;

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

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r

            // $unroll 1
            scalar colp0$i, colp1$i, colp2$i, colq0$i, colq1$i, colq2$i;
            scalar dist0$i, dist1$i, dist2$i;
            scalar sqdist0$i, sqdist1$i, sqdist2$i;
            scalar cvarp0$i, cvarp1$i, cvarp2$i, cvarq0$i, cvarq1$i, cvarq2$i;
            scalar var_cancel0$i, var_cancel1$i, var_cancel2$i;
            scalar normalization0$i, normalization1$i, normalization2$i;
            scalar term0$i, term1$i, term2$i;


            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; yp+=$n) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    colp0$i = c[0][xp][yp + $i];
                    colp1$i = c[1][xp][yp + $i];
                    colp2$i = c[2][xp][yp + $i];
                    colq0$i = c[0][xq][yq + $i];
                    colq1$i = c[1][xq][yq + $i];
                    colq2$i = c[2][xq][yq + $i];
                    cvarp0$i = c_var[0][xp][yp + $i];
                    cvarp1$i = c_var[1][xp][yp + $i];
                    cvarp2$i = c_var[2][xp][yp + $i];
                    cvarq0$i = c_var[0][xq][yq + $i];
                    cvarq1$i = c_var[1][xq][yq + $i];
                    cvarq2$i = c_var[2][xq][yq + $i];

                    dist0$i = colp0$i - colq0$i;
                    dist1$i = colp1$i - colq1$i;
                    dist2$i = colp2$i - colq2$i;

                    sqdist0$i = dist0$i * dist0$i;
                    sqdist1$i = dist1$i * dist1$i;
                    sqdist2$i = dist2$i * dist2$i;

                    var_cancel0$i = cvarp0$i + fmin(cvarp0$i, cvarq0$i);
                    var_cancel1$i = cvarp1$i + fmin(cvarp1$i, cvarq1$i);
                    var_cancel2$i = cvarp2$i + fmin(cvarp2$i, cvarq2$i);

                    normalization0$i = EPSILON + K_C_SQUARED*(cvarp0$i + cvarq0$i);
                    normalization1$i = EPSILON + K_C_SQUARED*(cvarp1$i + cvarq1$i);
                    normalization2$i = EPSILON + K_C_SQUARED*(cvarp2$i + cvarq2$i);
                    
                    term0$i = (sqdist0$i - var_cancel0$i) / normalization0$i;
                    term1$i = (sqdist1$i - var_cancel1$i) / normalization1$i;
                    term2$i = (sqdist2$i - var_cancel2$i) / normalization2$i;
                    
                    temp[xp * W + yp + $i] = term0$i + term1$i + term2$i;
                }
            }
            // $end_unroll


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            // $unroll 8
            scalar sum1_$i;
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=$n) {

                    scalar sum1_$i = 0.f;

                    for (int k=-F; k<=F; k++){
                        sum1_$i += temp[xp * W + yp+k+$i];
                    }

                    temp2[xp * W + yp + $i] = sum1_$i;
                }
            }
            // $end_unroll

            // (2) Convolve along width including weighted contribution
            // $unroll 4
            scalar sum2_$i, weight2_$i;
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=$n) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    sum2_$i = 0.f;

                    sum2_$i += temp2[(xp-3) * W + yp + $i];

                    sum2_$i += temp2[(xp-2) * W + yp + $i];

                    sum2_$i += temp2[(xp-1) * W + yp + $i];

                    sum2_$i += temp2[(xp) * W + yp + $i];

                    sum2_$i += temp2[(xp+1) * W + yp + $i];

                    sum2_$i += temp2[(xp+2) * W + yp + $i];

                    sum2_$i += temp2[(xp+3) * W + yp + $i];
                    
                    // Final Weight
                    weight2_$i = exp(-fmax(0.f, (sum2_$i * NEIGH_INV)));
                    
                    weight_sum[xp * W + yp + $i] += weight2_$i;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp+$i] += weight2_$i * input[i][xq][yq+$i];
                    }
                }
            }
            // $end_unroll

        }
    }

    // Final Weight Normalization
    // $unroll 4
    scalar w_$i;
    for(int xp = R + F; xp < W - R - F; ++xp) {
        for(int yp = R + F; yp < H - R - F; yp+=$n) {
        
            w_$i = weight_sum[xp * W + yp + $i];

            for (int i=0; i<3; i++){
                output[i][xp][yp+$i] /= w_$i;
            }
        }
    }
    // $end_unroll

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

void filtering_basic_f1_ILP(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H){
    

    scalar K_C_SQUARED = p.kc * p.kc;
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

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; ++yp) {

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
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=8) {

                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
                    scalar sum_4 = 0.f;
                    scalar sum_5 = 0.f;
                    scalar sum_6 = 0.f;
                    scalar sum_7 = 0.f;

                    for (int k=-F; k<=F; k++){
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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=4) {

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
        for(int yp = R + F; yp < H - R - F; yp+=4) {
        
            w_0 = weight_sum[xp * W + yp];
            w_1 = weight_sum[xp * W + yp+1];
            w_2 = weight_sum[xp * W + yp+2];
            w_3 = weight_sum[xp * W + yp+3];

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


void feature_prefiltering_ILP(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    scalar K_C_SQUARED = p.kc * p.kc;
    int R = p.r;
    int F = p.f;

    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar sqdist0, var_cancel0, normalization0, term0;
    scalar sqdist1, var_cancel1, normalization1, term1;
    scalar sqdist2, var_cancel2, normalization2, term2;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;

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
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R; yp < H - R; ++yp) {

                    xq = xp + r_x;
                    yq = yp + r_y;
                    
                    sqdist0 = features[0][xp][yp] - features[0][xq][yq];
                    sqdist1 = features[1][xp][yp] - features[1][xq][yq];
                    sqdist2 = features[2][xp][yp] - features[2][xq][yq];

                    sqdist0 *= sqdist0;
                    sqdist1 *= sqdist1;
                    sqdist2 *= sqdist2;

                    var_cancel0 = features_var[0][xp][yp] + fmin(features_var[0][xp][yp], features_var[0][xq][yq]);
                    var_cancel1 = features_var[1][xp][yp] + fmin(features_var[1][xp][yp], features_var[1][xq][yq]);
                    var_cancel2 = features_var[2][xp][yp] + fmin(features_var[2][xp][yp], features_var[2][xq][yq]);

                    normalization0 = EPSILON + K_C_SQUARED*(features_var[0][xp][yp] + features_var[0][xq][yq]);
                    normalization1 = EPSILON + K_C_SQUARED*(features_var[1][xp][yp] + features_var[1][xq][yq]);
                    normalization2 = EPSILON + K_C_SQUARED*(features_var[2][xp][yp] + features_var[2][xq][yq]);
                    
                    term0 = (sqdist0 - var_cancel0) / normalization0;
                    term1 = (sqdist1 - var_cancel1) / normalization1;
                    term2 = (sqdist2 - var_cancel2) / normalization2;
                    
                    temp[xp * W + yp] = term0 + term1 + term2;
                }
            }
            

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=8) {
                    
                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;
                    sum_4 = 0.f;
                    sum_5 = 0.f;
                    sum_6 = 0.f;
                    sum_7 = 0.f;

                    // Unrolled summation 
                    for (int k=-F; k<=F; k++){
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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=4) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;

                    // Unrolled summation => tailed to f=3 => 2*f+1 = 7
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

                    // Final weight computation
                    weight_0 = exp(-fmax(0.f, (sum_0 * NEIGH_INV)));
                    weight_1 = exp(-fmax(0.f, (sum_1 * NEIGH_INV)));
                    weight_2 = exp(-fmax(0.f, (sum_2 * NEIGH_INV)));
                    weight_3 = exp(-fmax(0.f, (sum_3 * NEIGH_INV)));

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
    for(int xp = R + F; xp < W - R - F; ++xp) {
        for(int yp = R + F; yp < H - R - F; yp+=4) {
        
            w_0 = weight_sum[xp * W + yp];
            w_1 = weight_sum[xp * W + yp+1];
            w_2 = weight_sum[xp * W + yp+2];
            w_3 = weight_sum[xp * W + yp+3];

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


    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F; yp++){
                output[i][xp][yp] = features[i][xp][yp];
                output[i][xp][H - yp - 1] = features[i][xp][H - yp - 1];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][xp][H - yp - 1] = features_var[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F; xp++){
            for (int yp = R + F ; yp < H - R - F; yp++){
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


void candidate_filtering_FIRST_ILP(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;

    scalar K_C_SQUARED = p.kc * p.kc;
    scalar K_F_SQUARED = p.kf * p.kf;
    int R = p.r;
    int F = p.f;

    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar diffL, diffR, diff, diffU, diffD;
    scalar sqdist, var_cancel, dist_var, var_term, var_max, normalization, df;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;


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
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F; x < W - R - F; ++x) {
            for(int y =  R + F; y < H -  R - F; ++y) {
                diffL = features[i][x][y] - features[i][x-1][y];
                diffR = features[i][x][y] - features[i][x+1][y];
                diffU = features[i][x][y] - features[i][x][y-1];
                diffD = features[i][x][y] - features[i][x][y+1];
                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;
                                          
                        sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                        normalization = EPSILON + K_C_SQUARED*(var_term);
                        dist_var = var_cancel - sqdist;
                        temp[xp * W + yp] += (dist_var / normalization);
                    }
                }
            }

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        df = feature_weights[xp * W + yp];

                        sqdist = features[j][xp][yp] - features[j][xq][yq];
                        var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        dist_var = var_cancel - sqdist;

                        var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        normalization = K_F_SQUARED*fmax(p.tau, var_max);

                        df = fmin(df, (dist_var)/normalization);
                        feature_weights[xp * W + yp] = df;
                    }
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=8) {

                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;
                    sum_4 = 0.f;
                    sum_5 = 0.f;
                    sum_6 = 0.f;
                    sum_7 = 0.f;

                    for (int k=-F; k<=F; k++){
                        sum_0 += temp[xp * W + yp + k];
                        sum_1 += temp[xp * W + yp + k + 1];
                        sum_2 += temp[xp * W + yp + k + 2];
                        sum_3 += temp[xp * W + yp + k + 3];
                        sum_4 += temp[xp * W + yp + k + 4];
                        sum_5 += temp[xp * W + yp + k + 5];
                        sum_6 += temp[xp * W + yp + k + 6];
                        sum_7 += temp[xp * W + yp + k + 7];
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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=4) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    // Compute final color weight
                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;

                    // Unrolled summation => tailed to f=1 => 2*f+1 = 3
                    sum_0 += temp2[(xp-1) * W + yp];
                    sum_1 += temp2[(xp-1) * W + yp + 1];
                    sum_2 += temp2[(xp-1) * W + yp + 2];
                    sum_3 += temp2[(xp-1) * W + yp + 3];

                    sum_0 += temp2[(xp) * W + yp];
                    sum_1 += temp2[(xp) * W + yp + 1];
                    sum_2 += temp2[(xp) * W + yp + 2];
                    sum_3 += temp2[(xp) * W + yp + 3];
                    
                    sum_0 += temp2[(xp+1) * W + yp];                    
                    sum_1 += temp2[(xp+1) * W + yp + 1];
                    sum_2 += temp2[(xp+1) * W + yp + 2];
                    sum_3 += temp2[(xp+1) * W + yp + 3];
                    
                    color_weight_0 = (sum_0 * NEIGH_INV);
                    color_weight_1 = (sum_1 * NEIGH_INV);
                    color_weight_2 = (sum_2 * NEIGH_INV);
                    color_weight_3 = (sum_3 * NEIGH_INV);
                    
                    // Final weight computation
                    weight_0 = exp(fmin(color_weight_0, feature_weights[xp * W + yp]));
                    weight_1 = exp(fmin(color_weight_1, feature_weights[xp * W + yp + 1]));
                    weight_2 = exp(fmin(color_weight_2, feature_weights[xp * W + yp + 2]));
                    weight_3 = exp(fmin(color_weight_3, feature_weights[xp * W + yp + 3]));

                    weight_sum[xp * W + yp] += weight_0;
                    weight_sum[xp * W + yp + 1] += weight_1;
                    weight_sum[xp * W + yp + 2] += weight_2;
                    weight_sum[xp * W + yp + 3] += weight_3;
                    
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
    for(int xp = R + F; xp < W - R - F; ++xp) {
        for(int yp = R + F; yp < H - R - F; yp+=4) {
        
            w_0 = weight_sum[xp * W + yp];
            w_1 = weight_sum[xp * W + yp+1];
            w_2 = weight_sum[xp * W + yp+2];
            w_3 = weight_sum[xp * W + yp+3];

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

void candidate_filtering_SECOND_ILP(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;

    scalar K_C_SQUARED = p.kc * p.kc;
    scalar K_F_SQUARED = p.kf * p.kf;
    int R = p.r;
    int F = p.f;

    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar diffL, diffR, diff, diffU, diffD;
    scalar sqdist, var_cancel, dist_var, var_term, var_max, normalization, df;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7;
    scalar w_0, w_1, w_2, w_3;


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
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F; x < W - R - F; ++x) {
            for(int y =  R + F; y < H -  R - F; ++y) {
                diffL = features[i][x][y] - features[i][x-1][y];
                diffR = features[i][x][y] - features[i][x+1][y];
                diffU = features[i][x][y] - features[i][x][y-1];
                diffD = features[i][x][y] - features[i][x][y+1];
                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;
                                          
                        sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                        normalization = EPSILON + K_C_SQUARED*(var_term);
                        dist_var = var_cancel - sqdist;
                        temp[xp * W + yp] += (dist_var / normalization);
                    }
                }
            }

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        df = feature_weights[xp * W + yp];

                        sqdist = features[j][xp][yp] - features[j][xq][yq];
                        var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        dist_var = var_cancel - sqdist;

                        var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        normalization = K_F_SQUARED*fmax(p.tau, var_max);

                        df = fmin(df, (dist_var)/normalization);
                        feature_weights[xp * W + yp] = df;
                    }
                }
            }


            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=8) {
                    
                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;
                    sum_4 = 0.f;
                    sum_5 = 0.f;
                    sum_6 = 0.f;
                    sum_7 = 0.f;

                    for (int k=-F; k<=F; k++){
                        sum_0 += temp[xp * W + yp + k];
                        sum_1 += temp[xp * W + yp + k + 1];
                        sum_2 += temp[xp * W + yp + k + 2];
                        sum_3 += temp[xp * W + yp + k + 3];
                        sum_4 += temp[xp * W + yp + k + 4];
                        sum_5 += temp[xp * W + yp + k + 5];
                        sum_6 += temp[xp * W + yp + k + 6];
                        sum_7 += temp[xp * W + yp + k + 7];
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
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;

                    // Unrolled summation => tailed to f=3 => 2*f+1 = 7
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
                    
                    // Final weight computation
                    color_weight_0 = (sum_0 * NEIGH_INV);
                    color_weight_1 = (sum_1 * NEIGH_INV);
                    color_weight_2 = (sum_2 * NEIGH_INV);
                    color_weight_3 = (sum_3 * NEIGH_INV);
                    
                    weight_0 = exp(fmin(color_weight_0, feature_weights[xp * W + yp]));
                    weight_1 = exp(fmin(color_weight_1, feature_weights[xp * W + yp + 1]));
                    weight_2 = exp(fmin(color_weight_2, feature_weights[xp * W + yp + 2]));
                    weight_3 = exp(fmin(color_weight_3, feature_weights[xp * W + yp + 3]));

                    weight_sum[xp * W + yp] += weight_0;
                    weight_sum[xp * W + yp + 1] += weight_1;
                    weight_sum[xp * W + yp + 2] += weight_2;
                    weight_sum[xp * W + yp + 3] += weight_3;
                    
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
    for(int xp = R + F; xp < W - R - F; ++xp) {
        for(int yp = R + F; yp < H - R - F; yp+=4) {
        
            w_0 = weight_sum[xp * W + yp];
            w_1 = weight_sum[xp * W + yp+1];
            w_2 = weight_sum[xp * W + yp+2];
            w_3 = weight_sum[xp * W + yp+3];

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

void candidate_filtering_THIRD_ILP(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    int WH = W * H;

    scalar K_C_SQUARED = p.kc * p.kc;
    scalar K_F_SQUARED = p.kf * p.kf;
    int R = p.r;
    int F = p.f;

    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar diffL, diffR, diff, diffU, diffD;
    scalar sqdist, var_cancel, dist_var, var_max, normalization, df;
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar w_0, w_1, w_2, w_3;

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
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F; x < W - R - F; ++x) {
            for(int y =  R + F; y < H -  R - F; ++y) {
                diffL = features[i][x][y] - features[i][x-1][y];
                diffR = features[i][x][y] - features[i][x+1][y];
                diffU = features[i][x][y] - features[i][x][y-1];
                diffD = features[i][x][y] - features[i][x][y+1];
                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        

            // Compute features
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R + F; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        df = feature_weights[xp * W + yp];

                        sqdist = features[j][xp][yp] - features[j][xq][yq];
                        var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        dist_var = var_cancel - sqdist;

                        var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        normalization = K_F_SQUARED*fmax(p.tau, var_max);

                        df = fmin(df, (dist_var)/normalization);
                        feature_weights[xp * W + yp] = df;
                    }
                }
            }


            // (2) Convolve along width including weighted contribution
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=4) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    weight_0 = exp(feature_weights[xp * W + yp]);
                    weight_1 = exp(feature_weights[xp * W + yp+1]);
                    weight_2 = exp(feature_weights[xp * W + yp+2]);
                    weight_3 = exp(feature_weights[xp * W + yp+3]);

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
    for(int xp = R + F; xp < W - R - F; ++xp) {
        for(int yp = R + F; yp < H - R - F; yp+=4) {
        
            w_0 = weight_sum[xp * W + yp];
            w_1 = weight_sum[xp * W + yp+1];
            w_2 = weight_sum[xp * W + yp+2];
            w_3 = weight_sum[xp * W + yp+3];

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



void candidate_filtering_all_ILP(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int W, int H){

    int WH = W*H;

    int F_R = p[0].f;
    int F_G = p[1].f;
    int F_B = p[2].f;
    scalar TAU_R = p[0].tau;
    scalar TAU_G = p[1].tau;
    scalar TAU_B = p[2].tau;
    scalar K_C_SQUARED_R = p[0].kc * p[0].kc;
    scalar K_F_SQUARED_R = p[0].kf * p[0].kf;
    scalar K_F_SQUARED_B = p[2].kf * p[2].kf;

    int F_MIN = fmin(F_R, fmin(F_G, F_B));
    int R = p[0].r;

    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
    int xq, yq;
    scalar diffL, diffR, diff, diffU, diffD;
    scalar sqdist, var_cancel, var_term, dist_var, var_max, normalization_r, normalization_b, df_r, df_b;
    scalar sum_r_0, sum_r_1, sum_r_2, sum_r_3, sum_r_4, sum_r_5, sum_r_6, sum_r_7;
    scalar sum_g_0, sum_g_1, sum_g_2, sum_g_3, sum_g_4, sum_g_5, sum_g_6, sum_g_7;
    scalar sum_0, sum_1, sum_2, sum_3;
    scalar color_weight_0, color_weight_1, color_weight_2, color_weight_3; 
    scalar weight_0, weight_1, weight_2, weight_3;
    scalar w_0, w_1, w_2, w_3;

    // ------------------------
    // MEMORY ALLOCATION
    // ------------------------

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
    
    // ------------------------
    // GRADIENT COMPUTATION
    // ------------------------

    // Compute gradients
    scalar *gradients;
    gradients = (scalar*) malloc(3 * W * H * sizeof(scalar));

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+F_MIN; x < W - R - F_MIN; ++x) {
            for(int y =  R+F_MIN; y < H -  R - F_MIN; ++y) {
                
                diffL = features[i][x][y] - features[i][x-1][y];
                diffR = features[i][x][y] - features[i][x+1][y];
                diffU = features[i][x][y] - features[i][x][y-1];
                diffD = features[i][x][y] - features[i][x][y+1];

                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            
            } 
        }
    }

    // Precompute size of neighbourhood
    scalar NEIGH_R_INV = 1. / (3*(2*F_R+1)*(2*F_R+1));
    scalar NEIGH_G_INV = 1. / (3*(2*F_G+1)*(2*F_G+1));

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

                    xq = xp + r_x;
                    yq = yp + r_y;   
                    
                    sqdist = color[i][xp][yp] - color[i][xq][yq];
                    sqdist *= sqdist;
                    var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                    var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                    normalization_r = EPSILON + K_C_SQUARED_R*(var_term);
                    dist_var = var_cancel - sqdist;
                    temp[xp * W + yp] += (dist_var / normalization_r);

                    }
                }
            }

            // Precompute feature weights
            memset(features_weights_r, 0, W*H*sizeof(scalar));
            memset(features_weights_b, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F_MIN; xp < W - R - F_MIN; ++xp) {
                    for(int yp = R + F_MIN; yp < H - R - F_MIN; ++yp) {
                        
                        xq = xp + r_x;
                        yq = yp + r_y;

                        df_r = features_weights_r[xp * W + yp];
                        df_b = features_weights_b[xp * W + yp];
          
                        sqdist = features[j][xp][yp] - features[j][xq][yq];
                        var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        sqdist *= sqdist;
                        scalar dist_var = var_cancel - sqdist;

                        var_max = fmax(features_var[j][xp][yp], gradients[j * WH + xp * W + yp]);
                        normalization_r = K_F_SQUARED_R*fmax(TAU_R, var_max);
                        normalization_b = K_F_SQUARED_B*fmax(TAU_B, var_max);

                        df_r = fmin(df_r, (dist_var)/normalization_r);
                        df_b = fmin(df_b, (dist_var)/normalization_b);
                        
                        features_weights_r[xp * W + yp] = df_r;
                        features_weights_b[xp * W + yp] = df_b;
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
                for(int yp = R + F_R; yp < H - R - F_R; yp+=8) {
                    
                    sum_r_0 = 0.f;
                    sum_r_1 = 0.f;
                    sum_r_2 = 0.f;
                    sum_r_3 = 0.f;
                    sum_r_4 = 0.f;
                    sum_r_5 = 0.f;
                    sum_r_6 = 0.f;
                    sum_r_7 = 0.f;

                    for (int k=-F_R; k<=F_R; k++){
                        sum_r_0 += temp[xp * W + yp+k+0];
                        sum_r_1 += temp[xp * W + yp+k+1];
                        sum_r_2 += temp[xp * W + yp+k+2];
                        sum_r_3 += temp[xp * W + yp+k+3];
                        sum_r_4 += temp[xp * W + yp+k+4];
                        sum_r_5 += temp[xp * W + yp+k+5];
                        sum_r_6 += temp[xp * W + yp+k+6];
                        sum_r_7 += temp[xp * W + yp+k+7];
                    }
                    temp2_r[xp * W + yp+0] = sum_r_0;
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
            for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;

                    // Unrolled Summation => Fixed for F_R=1 => 2*f+1 = 3
                    sum_0 += temp2_r[(xp-1) * W + yp+0];
                    sum_1 += temp2_r[(xp-1) * W + yp+1];    
                    sum_2 += temp2_r[(xp-1) * W + yp+2];    
                    sum_3 += temp2_r[(xp-1) * W + yp+3];

                    sum_0 += temp2_r[(xp) * W + yp+0];
                    sum_1 += temp2_r[(xp) * W + yp+1];
                    sum_2 += temp2_r[(xp) * W + yp+2];
                    sum_3 += temp2_r[(xp) * W + yp+3];

                    sum_0 += temp2_r[(xp+1) * W + yp+0];
                    sum_1 += temp2_r[(xp+1) * W + yp+1];
                    sum_2 += temp2_r[(xp+1) * W + yp+2];
                    sum_3 += temp2_r[(xp+1) * W + yp+3];


                    // Compute final weight
                    weight_0 = exp(fmin((sum_0 * NEIGH_R_INV), features_weights_r[xp * W + yp + 0]));
                    weight_1 = exp(fmin((sum_1 * NEIGH_R_INV), features_weights_r[xp * W + yp + 1]));
                    weight_2 = exp(fmin((sum_2 * NEIGH_R_INV), features_weights_r[xp * W + yp + 2]));
                    weight_3 = exp(fmin((sum_3 * NEIGH_R_INV), features_weights_r[xp * W + yp + 3]));

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
            // (1) Convolve along height
            
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=8) {
                    
                    sum_g_0 = 0.f;
                    sum_g_1 = 0.f;
                    sum_g_2 = 0.f;
                    sum_g_3 = 0.f;
                    sum_g_4 = 0.f;
                    sum_g_5 = 0.f;
                    sum_g_6 = 0.f;
                    sum_g_7 = 0.f;

                    for (int k=-F_G; k<=F_G; k++){
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
            for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    sum_0 = 0.f;
                    sum_1 = 0.f;
                    sum_2 = 0.f;
                    sum_3 = 0.f;

                    // Unrolled Summation => Fixed for F_G=3 => 2*F_G+1 = 7
                    sum_0 += temp2_g[(xp-3) * W + yp];
                    sum_1 += temp2_g[(xp-3) * W + yp+1];
                    sum_2 += temp2_g[(xp-3) * W + yp+2];
                    sum_3 += temp2_g[(xp-3) * W + yp+3];

                    sum_0 += temp2_g[(xp-2) * W + yp];
                    sum_1 += temp2_g[(xp-2) * W + yp+1];
                    sum_2 += temp2_g[(xp-2) * W + yp+2];
                    sum_3 += temp2_g[(xp-2) * W + yp+3];

                    sum_0 += temp2_g[(xp-1) * W + yp];
                    sum_1 += temp2_g[(xp-1) * W + yp+1];
                    sum_2 += temp2_g[(xp-1) * W + yp+2];
                    sum_3 += temp2_g[(xp-1) * W + yp+3];

                    sum_0 += temp2_g[(xp) * W + yp];
                    sum_1 += temp2_g[(xp) * W + yp+1];
                    sum_2 += temp2_g[(xp) * W + yp+2];
                    sum_3 += temp2_g[(xp) * W + yp+3];

                    sum_0 += temp2_g[(xp+1) * W + yp];
                    sum_1 += temp2_g[(xp+1) * W + yp+1];
                    sum_2 += temp2_g[(xp+1) * W + yp+2];
                    sum_3 += temp2_g[(xp+1) * W + yp+3];
                    
                    sum_0 += temp2_g[(xp+2) * W + yp];
                    sum_1 += temp2_g[(xp+2) * W + yp+1];
                    sum_2 += temp2_g[(xp+2) * W + yp+2];
                    sum_3 += temp2_g[(xp+2) * W + yp+3];

                    sum_0 += temp2_g[(xp+3) * W + yp];
                    sum_1 += temp2_g[(xp+3) * W + yp+1];
                    sum_2 += temp2_g[(xp+3) * W + yp+2];
                    sum_3 += temp2_g[(xp+3) * W + yp+3];
                    
                    // Compute final weight
                    weight_0 = exp(fmin((sum_0 * NEIGH_G_INV), features_weights_r[xp * W + yp]));
                    weight_1 = exp(fmin((sum_1 * NEIGH_G_INV), features_weights_r[xp * W + yp+1]));
                    weight_2 = exp(fmin((sum_2 * NEIGH_G_INV), features_weights_r[xp * W + yp+2]));
                    weight_3 = exp(fmin((sum_3 * NEIGH_G_INV), features_weights_r[xp * W + yp+3]));

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

            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    weight_0 = exp(features_weights_b[xp * W + yp]);
                    weight_1 = exp(features_weights_b[xp * W + yp+1]);
                    weight_2 = exp(features_weights_b[xp * W + yp+2]);
                    weight_3 = exp(features_weights_b[xp * W + yp+3]);

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
        for(int yp = R + F_R; yp < H - R - F_R; yp+=4) {
        
            w_0 = weight_sum_r[xp * W + yp];
            w_1 = weight_sum_r[xp * W + yp+1];
            w_2 = weight_sum_r[xp * W + yp+2];
            w_3 = weight_sum_r[xp * W + yp+3];

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
        for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {
        
            w_0 = weight_sum_g[xp * W + yp];
            w_1 = weight_sum_g[xp * W + yp+1];
            w_2 = weight_sum_g[xp * W + yp+2];
            w_3 = weight_sum_g[xp * W + yp+3];

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
        for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {
        
            w_0 = weight_sum_b[xp * W + yp];
            w_1 = weight_sum_b[xp * W + yp+1];
            w_2 = weight_sum_b[xp * W + yp+2];
            w_3 = weight_sum_b[xp * W + yp+3];

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
    // Candidate FIRST and THIRD (due to F_R = F_B)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F_R; yp++){
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F_R; xp++){
            for (int yp = R + F_R ; yp < H - R - F_R; yp++){
            
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][W - xp - 1][yp] = color[i][W - xp - 1][yp];
             }
        }
    }

    // Candidate SECOND since F_G != F_R
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + F_G; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][xp][H - yp - 1] = color[i][xp][H - yp - 1];
            }
        }
        for(int xp = 0; xp < R + F_G; xp++){
            for (int yp = R + F_G ; yp < H - R - F_G; yp++){
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
    free(temp2_g);
    free(features_weights_r);
    free(features_weights_b);
    free(gradients);

}

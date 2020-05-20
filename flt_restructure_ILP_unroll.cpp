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

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    sum2_$i = 0.f;

                    // Unrolled summation => tailed to f=1 => 2*f+1 = 3
                    sum2_$i += temp2[(xp-1) * W + yp+$i];

                    sum2_$i += temp2[(xp) * W + yp+$i];

                    sum2_$i += temp2[(xp+1) * W + yp+$i];
                    
                    weight2_$i = exp(-fmax(0.f, (sum2_$i * NEIGH_INV)));
                    
                    weight_sum[xp * W + yp+$i] += weight2_$i;
                    
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


void feature_prefiltering_ILP(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H){

    scalar K_C_SQUARED = p.kc * p.kc;
    int R = p.r;
    int F = p.f;

    // ------------------------
    // VARIABLE DEFINITION
    // ------------------------
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
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));

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

                    colp0$i = features[0][xp][yp + $i];
                    colp1$i = features[1][xp][yp + $i];
                    colp2$i = features[2][xp][yp + $i];
                    colq0$i = features[0][xq][yq + $i];
                    colq1$i = features[1][xq][yq + $i];
                    colq2$i = features[2][xq][yq + $i];
                    cvarp0$i = features_var[0][xp][yp + $i];
                    cvarp1$i = features_var[1][xp][yp + $i];
                    cvarp2$i = features_var[2][xp][yp + $i];
                    cvarq0$i = features_var[0][xq][yq + $i];
                    cvarq1$i = features_var[1][xq][yq + $i];
                    cvarq2$i = features_var[2][xq][yq + $i];

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

                    // Unrolled summation => tailed to f=3 => 2*f+1 = 7
                    sum2_$i += temp2[(xp-3) * W + yp + $i];

                    sum2_$i += temp2[(xp-2) * W + yp + $i];

                    sum2_$i += temp2[(xp-1) * W + yp + $i];

                    sum2_$i += temp2[(xp) * W + yp + $i];

                    sum2_$i += temp2[(xp+1) * W + yp + $i];

                    sum2_$i += temp2[(xp+2) * W + yp + $i];

                    sum2_$i += temp2[(xp+3) * W + yp + $i];

                    // Final weight computation
                    weight2_$i = exp(-fmax(0.f, (sum2_$i * NEIGH_INV)));
                    
                    weight_sum[xp * W + yp + $i] += weight2_$i;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp+$i] += weight2_$i * features[i][xq][yq+$i];
                        output_var[i][xp][yp+$i] += weight2_$i * features_var[i][xq][yq+$i];
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
                output_var[i][xp][yp+$i] /= w_$i;
            }
        }
    }
    // $end_unroll

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

    // $unroll 1
    scalar featU$i, featD$i, featR$i, featL$i, feat$i;
    scalar diffU$i, diffD$i, diffR$i, diffL$i;
    scalar sqdiffU$i, sqdiffD$i, sqdiffR$i, sqdiffL$i;
    scalar gradH$i, gradV$i;
    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F; x < W - R - F; ++x) {
            for(int y =  R + F; y < H -  R - F; y+=$n) {
                featU$i = features[i][x][y+$i-1];
                featD$i = features[i][x][y+$i+1];
                featL$i = features[i][x-1][y+$i];
                featR$i = features[i][x+1][y+$i];
                feat$i = features[i][x][y+$i];

                diffL$i = feat$i - featL$i;
                diffR$i = feat$i - featR$i;
                diffU$i = feat$i - featU$i;
                diffD$i = feat$i - featD$i;

                sqdiffL$i = diffL$i * diffL$i;
                sqdiffR$i = diffR$i * diffR$i;
                sqdiffU$i = diffU$i * diffU$i;
                sqdiffD$i = diffD$i * diffD$i;

                gradH$i = fmin(sqdiffL$i, sqdiffR$i);
                gradV$i = fmin(sqdiffU$i, sqdiffD$i);
                
                gradients[i * WH + x * W + y+$i] = gradH$i + gradV$i;
            }
        } 
    }
    // $end_unroll

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            // $unroll 1
            scalar colp$i, colq$i, dist$i, sqdist$i, cvarp$i, cvarq$i, var_cancel$i, normalization$i, var_term$i, dist_var$i;
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        colp$i = color[i][xp][yp + $i];
                        colq$i = color[i][xq][yq + $i];
                        cvarp$i = color_var[i][xp][yp + $i];
                        cvarq$i = color_var[i][xq][yq + $i];

                        dist$i = colp$i - colq$i;

                        sqdist$i = dist$i * dist$i;

                        var_cancel$i = cvarp$i + fmin(cvarp$i, cvarq$i);

                        var_term$i = cvarp$i + cvarq$i;

                        normalization$i = EPSILON + K_C_SQUARED*var_term$i;

                        dist_var$i = var_cancel$i - sqdist$i;

                        temp[xp * W + yp + $i] += (dist_var$i / normalization$i);
                    }
                }
            }
            // $end_unroll

            // Compute features
            // $unroll 1
            scalar df$i, featp$i, featq$i, distdf$i, sqdistdf$i, var_canceldf$i, fvarp$i, fvarq$i, dist_vardf$i, var_max$i, normalizationf$i;
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; yp += $n) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        df$i = feature_weights[xp * W + yp + $i];
                        featp$i = features[j][xp][yp + $i];
                        featq$i = features[j][xq][yq + $i];
                        fvarp$i = features_var[j][xp][yp + $i];
                        fvarq$i = features_var[j][xq][yq + $i];

                        distdf$i = featp$i - featq$i;
                        var_canceldf$i = fvarp$i + fmin(fvarp$i, fvarq$i);
                        sqdistdf$i = distdf$i * distdf$i;
                        dist_vardf$i = var_canceldf$i - sqdistdf$i;

                        var_max$i = fmax(fvarp$i, gradients[j * WH + xp * W + yp + $i]);
                        scalar normalizationf$i = K_F_SQUARED*fmax(p.tau, var_max$i);

                        df$i = fmin(df$i, (dist_vardf$i)/normalizationf$i);
                        feature_weights[xp * W + yp + $i] = df$i;
                    }
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
            scalar sum2_$i, color_weight_$i, weight2_$i;
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=$n) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    // Compute final color weight
                    sum2_$i = 0.f;

                    // Unrolled summation => tailed to f=1 => 2*f+1 = 3
                    sum2_$i += temp2[(xp-1) * W + yp+$i];

                    sum2_$i += temp2[(xp) * W + yp+$i];

                    sum2_$i += temp2[(xp+1) * W + yp+$i];
                    
                    color_weight_$i = (sum2_$i * NEIGH_INV);
                    
                    // Final weight computation
                    weight2_$i = exp(fmin(color_weight_$i, feature_weights[xp * W + yp + $i]));

                    weight_sum[xp * W + yp+$i] += weight2_$i;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp+$i] += weight2_$i * color[i][xq][yq+$i];
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

    // $unroll 1
    scalar featU$i, featD$i, featR$i, featL$i, feat$i;
    scalar diffU$i, diffD$i, diffR$i, diffL$i;
    scalar sqdiffU$i, sqdiffD$i, sqdiffR$i, sqdiffL$i;
    scalar gradH$i, gradV$i;
    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F; x < W - R - F; ++x) {
            for(int y =  R + F; y < H -  R - F; y+=$n) {
                featU$i = features[i][x][y+$i-1];
                featD$i = features[i][x][y+$i+1];
                featL$i = features[i][x-1][y+$i];
                featR$i = features[i][x+1][y+$i];
                feat$i = features[i][x][y+$i];

                diffL$i = feat$i - featL$i;
                diffR$i = feat$i - featR$i;
                diffU$i = feat$i - featU$i;
                diffD$i = feat$i - featD$i;

                sqdiffL$i = diffL$i * diffL$i;
                sqdiffR$i = diffR$i * diffR$i;
                sqdiffU$i = diffU$i * diffU$i;
                sqdiffD$i = diffD$i * diffD$i;

                gradH$i = fmin(sqdiffL$i, sqdiffR$i);
                gradV$i = fmin(sqdiffU$i, sqdiffD$i);
                
                gradients[i * WH + x * W + y+$i] = gradH$i + gradV$i;
            }
        } 
    }
    // $end_unroll

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            // $unroll 1
            scalar colp$i, colq$i, dist$i, sqdist$i, cvarp$i, cvarq$i, var_cancel$i, normalization$i, var_term$i, dist_var$i;
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; ++yp) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        colp$i = color[i][xp][yp + $i];
                        colq$i = color[i][xq][yq + $i];
                        cvarp$i = color_var[i][xp][yp + $i];
                        cvarq$i = color_var[i][xq][yq + $i];

                        dist$i = colp$i - colq$i;

                        sqdist$i = dist$i * dist$i;

                        var_cancel$i = cvarp$i + fmin(cvarp$i, cvarq$i);

                        var_term$i = cvarp$i + cvarq$i;

                        normalization$i = EPSILON + K_C_SQUARED*var_term$i;

                        dist_var$i = var_cancel$i - sqdist$i;

                        temp[xp * W + yp + $i] += (dist_var$i / normalization$i);
                    }
                }
            }
            // $end_unroll

            // Compute features
            // $unroll 1
            scalar df$i, featp$i, featq$i, distdf$i, sqdistdf$i, var_canceldf$i, fvarp$i, fvarq$i, dist_vardf$i, var_max$i, normalizationf$i;
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; yp += $n) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        df$i = feature_weights[xp * W + yp + $i];
                        featp$i = features[j][xp][yp + $i];
                        featq$i = features[j][xq][yq + $i];
                        fvarp$i = features_var[j][xp][yp + $i];
                        fvarq$i = features_var[j][xq][yq + $i];

                        distdf$i = featp$i - featq$i;
                        var_canceldf$i = fvarp$i + fmin(fvarp$i, fvarq$i);
                        sqdistdf$i = distdf$i * distdf$i;
                        dist_vardf$i = var_canceldf$i - sqdistdf$i;

                        var_max$i = fmax(fvarp$i, gradients[j * WH + xp * W + yp + $i]);
                        scalar normalizationf$i = K_F_SQUARED*fmax(p.tau, var_max$i);

                        df$i = fmin(df$i, (dist_vardf$i)/normalizationf$i);
                        feature_weights[xp * W + yp + $i] = df$i;
                    }
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
            scalar sum2_$i, color_weight_$i, weight2_$i;
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=$n) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    sum2_$i = 0.f;

                    // Unrolled summation => tailed to f=3 => 2*f+1 = 7
                    sum2_$i += temp2[(xp-3) * W + yp + $i];

                    sum2_$i += temp2[(xp-2) * W + yp + $i];

                    sum2_$i += temp2[(xp-1) * W + yp + $i];

                    sum2_$i += temp2[(xp) * W + yp + $i];

                    sum2_$i += temp2[(xp+1) * W + yp + $i];

                    sum2_$i += temp2[(xp+2) * W + yp + $i];

                    sum2_$i += temp2[(xp+3) * W + yp + $i];
                    
                    // Final weight computation
                    color_weight_$i = (sum2_$i * NEIGH_INV);
                    
                    weight2_$i = exp(fmin(color_weight_$i, feature_weights[xp * W + yp + $i]));

                    weight_sum[xp * W + yp+$i] += weight2_$i;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp+$i] += weight2_$i * color[i][xq][yq+$i];
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

    // $unroll 1
    scalar featU$i, featD$i, featR$i, featL$i, feat$i;
    scalar diffU$i, diffD$i, diffR$i, diffL$i;
    scalar sqdiffU$i, sqdiffD$i, sqdiffR$i, sqdiffL$i;
    scalar gradH$i, gradV$i;
    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F; x < W - R - F; ++x) {
            for(int y =  R + F; y < H -  R - F; y+=$n) {
                featU$i = features[i][x][y+$i-1];
                featD$i = features[i][x][y+$i+1];
                featL$i = features[i][x-1][y+$i];
                featR$i = features[i][x+1][y+$i];
                feat$i = features[i][x][y+$i];

                diffL$i = feat$i - featL$i;
                diffR$i = feat$i - featR$i;
                diffU$i = feat$i - featU$i;
                diffD$i = feat$i - featD$i;

                sqdiffL$i = diffL$i * diffL$i;
                sqdiffR$i = diffR$i * diffR$i;
                sqdiffU$i = diffU$i * diffU$i;
                sqdiffD$i = diffD$i * diffD$i;

                gradH$i = fmin(sqdiffL$i, sqdiffR$i);
                gradV$i = fmin(sqdiffU$i, sqdiffD$i);
                
                gradients[i * WH + x * W + y+$i] = gradH$i + gradV$i;
            }
        } 
    }
    // $end_unroll

    // Precompute size of neighbourhood
    scalar NEIGH_INV = 1. / (3*(2*F+1)*(2*F+1));


    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        

            // Compute features
            // $unroll 1
            scalar df$i, featp$i, featq$i, distdf$i, sqdistdf$i, var_canceldf$i, fvarp$i, fvarq$i, dist_vardf$i, var_max$i, normalizationf$i;
            memset(feature_weights, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F; xp < W - R - F; ++xp) {
                    for(int yp = R + F; yp < H - R - F; yp += $n) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        df$i = feature_weights[xp * W + yp + $i];
                        featp$i = features[j][xp][yp + $i];
                        featq$i = features[j][xq][yq + $i];
                        fvarp$i = features_var[j][xp][yp + $i];
                        fvarq$i = features_var[j][xq][yq + $i];

                        distdf$i = featp$i - featq$i;
                        var_canceldf$i = fvarp$i + fmin(fvarp$i, fvarq$i);
                        sqdistdf$i = distdf$i * distdf$i;
                        dist_vardf$i = var_canceldf$i - sqdistdf$i;

                        var_max$i = fmax(fvarp$i, gradients[j * WH + xp * W + yp + $i]);
                        scalar normalizationf$i = K_F_SQUARED*fmax(p.tau, var_max$i);

                        df$i = fmin(df$i, (dist_vardf$i)/normalizationf$i);
                        feature_weights[xp * W + yp + $i] = df$i;
                    }
                }
            }
            // $end_unroll


            // (2) Convolve along width including weighted contribution
            // $unroll 4
            scalar weight2_$i;
            for(int xp = R + F; xp < W - R - F; ++xp) {
                for(int yp = R + F; yp < H - R - F; yp+=$n) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    weight2_$i = exp(feature_weights[xp * W + yp + $i]);

                    weight_sum[xp * W + yp + $i] += weight2_$i;

                    for (int i=0; i<3; i++){
                        output[i][xp][yp + $i] += weight2_$i * color[i][xq][yq + $i];
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

    // $unroll 1
    scalar featU$i, featD$i, featR$i, featL$i, feat$i;
    scalar diffU$i, diffD$i, diffR$i, diffL$i;
    scalar sqdiffU$i, sqdiffD$i, sqdiffR$i, sqdiffL$i;
    scalar gradH$i, gradV$i;
    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R + F_MIN; x < W - R - F_MIN; ++x) {
            for(int y =  R + F_MIN; y < H -  R - F_MIN; y+=$n) {
                featU$i = features[i][x][y+$i-1];
                featD$i = features[i][x][y+$i+1];
                featL$i = features[i][x-1][y+$i];
                featR$i = features[i][x+1][y+$i];
                feat$i = features[i][x][y+$i];

                diffL$i = feat$i - featL$i;
                diffR$i = feat$i - featR$i;
                diffU$i = feat$i - featU$i;
                diffD$i = feat$i - featD$i;

                sqdiffL$i = diffL$i * diffL$i;
                sqdiffR$i = diffR$i * diffR$i;
                sqdiffU$i = diffU$i * diffU$i;
                sqdiffD$i = diffD$i * diffD$i;

                gradH$i = fmin(sqdiffL$i, sqdiffR$i);
                gradV$i = fmin(sqdiffU$i, sqdiffD$i);
                
                gradients[i * WH + x * W + y+$i] = gradH$i + gradV$i;
            }
        } 
    }
    // $end_unroll

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
            // $unroll 1
            scalar colp$i, colq$i, dist$i, sqdist$i, cvarp$i, cvarq$i, var_cancel$i, normalization$i, var_term$i, dist_var$i;
            memset(temp, 0, W*H*sizeof(scalar));
            for (int i=0; i<3; i++){  
                for(int xp = R; xp < W - R; ++xp) {
                    for(int yp = R; yp < H - R; yp+=$n) {

                        xq = xp + r_x;
                        yq = yp + r_y;

                        colp$i = color[i][xp][yp + $i];
                        colq$i = color[i][xq][yq + $i];
                        cvarp$i = color_var[i][xp][yp + $i];
                        cvarq$i = color_var[i][xq][yq + $i];

                        dist$i = colp$i - colq$i;

                        sqdist$i = dist$i * dist$i;

                        var_cancel$i = cvarp$i + fmin(cvarp$i, cvarq$i);

                        var_term$i = cvarp$i + cvarq$i;

                        normalization$i = EPSILON + K_C_SQUARED_R*var_term$i;

                        dist_var$i = var_cancel$i - sqdist$i;

                        temp[xp * W + yp + $i] += (dist_var$i / normalization$i);
                    }
                }
            }
            // $end_unroll

            // Precompute feature weights
            // $unroll 1
            scalar df_r$i, df_b$i, featp$i, featq$i, distdf$i, sqdistdf$i, var_canceldf$i, fvarp$i, fvarq$i, dist_vardf$i, var_max$i, normalization_r$i, normalization_b$i;

            memset(features_weights_r, 0, W*H*sizeof(scalar));
            memset(features_weights_b, 0, W*H*sizeof(scalar));
            for(int j=0; j<NB_FEATURES;++j){
                for(int xp = R + F_MIN; xp < W - R - F_MIN; ++xp) {
                    for(int yp = R + F_MIN; yp < H - R - F_MIN; yp += $n) {
                        
                        xq = xp + r_x;
                        yq = yp + r_y;

                        df_r$i = features_weights_r[xp * W + yp + $i];
                        df_b$i = features_weights_b[xp * W + yp + $i];

                        featp$i = features[j][xp][yp + $i];
                        featq$i = features[j][xq][yq + $i];
                        fvarp$i = features_var[j][xp][yp + $i];
                        fvarq$i = features_var[j][xq][yq + $i];
          
                        distdf$i = featp$i - featq$i;
                        var_canceldf$i = fvarp$i + fmin(fvarp$i, fvarq$i);
                        sqdistdf$i = distdf$i * distdf$i;
                        dist_vardf$i = var_canceldf$i - sqdistdf$i;

                        var_max$i = fmax(fvarp$i, gradients[j * WH + xp * W + yp + $i]);
                        normalization_r$i = K_F_SQUARED_R*fmax(TAU_R, var_max$i);
                        normalization_b$i = K_F_SQUARED_B*fmax(TAU_B, var_max$i);

                        df_r$i = fmin(df_r$i, (dist_var$i)/normalization_r$i);
                        df_b$i = fmin(df_b$i, (dist_var$i)/normalization_b$i);
                        
                        features_weights_r[xp * W + yp + $i] = df_r$i;
                        features_weights_b[xp * W + yp + $i] = df_b$i;
                    } 
                }
            }
            // $end_unroll

            // #######################################################################################
            // BOX FILTERING => seperability of box filter kernel => two linear operations
            // #######################################################################################
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height

            // $unroll 8
            scalar sum_r_$i;
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=$n) {

                    scalar sum_r_$i = 0.f;

                    for (int k=-F_R; k<=F_R; k++){
                        sum_r_$i += temp[xp * W + yp+k+$i];
                    }

                    temp2_r[xp * W + yp + $i] = sum_r_$i;
                }
            }
            // $end_unroll

            // (2) Convolve along width including weighted contribution
            // $unroll 4
            scalar sum2_$i, weight2_$i;
            for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=$n) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    sum2_$i = 0.f;

                    // Unrolled summation => tailed to f=1 => 2*f+1 = 3
                    sum2_$i += temp2_r[(xp-1) * W + yp+$i];

                    sum2_$i += temp2_r[(xp) * W + yp+$i];

                    sum2_$i += temp2_r[(xp+1) * W + yp+$i];


                    // Compute final weight
                    weight2_$i = exp(fmin((sum2_$i * NEIGH_R_INV), features_weights_r[xp * W + yp + $i]));
                    
                    weight_sum_r[xp * W + yp+$i] += weight2_$i;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp+$i] += weight2_$i * color[i][xq][yq+$i];
                    }
                }
            }
            // $end_unroll


            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            
            // $unroll 8
            scalar sum_g_$i;
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=$n) {

                    scalar sum_g_$i = 0.f;

                    for (int k=-F_G; k<=F_G; k++){
                        sum_g_$i += temp[xp * W + yp+k+$i];
                    }

                    temp2_g[xp * W + yp + $i] = sum_g_$i;
                }
            }
            // $end_unroll

            // (2) Convolve along width including weighted contribution
            // $unroll 4
            scalar sumg_$i, weightg_$i;
            for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
                for(int yp = R + F_G; yp < H - R - F_G; yp+=$n) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    sumg_$i = 0.f;

                    // Unrolled summation => tailed to f=3 => 2*f+1 = 7
                    sumg_$i += temp2_g[(xp-3) * W + yp + $i];

                    sumg_$i += temp2_g[(xp-2) * W + yp + $i];

                    sumg_$i += temp2_g[(xp-1) * W + yp + $i];

                    sumg_$i += temp2_g[(xp) * W + yp + $i];

                    sumg_$i += temp2_g[(xp+1) * W + yp + $i];

                    sumg_$i += temp2_g[(xp+2) * W + yp + $i];

                    sumg_$i += temp2_g[(xp+3) * W + yp + $i];
                    
                    // Compute final weight
                    weightg_$i = exp(fmin((sumg_$i * NEIGH_G_INV), features_weights_r[xp * W + yp + $i]));

                    weight_sum_g[xp * W + yp + $i] += weightg_$i;
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp + $i] += weightg_$i * color[i][xq][yq+$i];
                    }
                }
            }
            // $end_unroll

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            // $unroll 4
            scalar weightb_$i;
            for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
                for(int yp = R + F_B; yp < H - R - F_B; yp+=$n) {

                    xq = xp + r_x;
                    yq = yp + r_y;

                    weightb_$i = exp(features_weights_b[xp * W + yp + $i]);

                    weight_sum_b[xp * W + yp + $i] += weightb_$i;

                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp+$i] += weightb_$i * color[i][xq][yq+$i];
                    }
                }
            }
            // $end_unroll

        }
    }


    // Final Weight Normalization R
    // $unroll 4
    scalar wr_$i;
    for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
        for(int yp = R + F_R; yp < H - R - F_R; yp+=$n) {
        
            wr_$i = weight_sum_r[xp * W + yp + $i];

            for (int i=0; i<3; i++){
                output_r[i][xp][yp+$i] /= wr_$i;
            }
        }
    }
    // $end_unroll

    // Final Weight Normalization G
    // $unroll 4
    scalar wg_$i;
    for(int xp = R + F_G; xp < W - R - F_G; ++xp) {
        for(int yp = R + F_G; yp < H - R - F_G; yp+=$n) {
        
            wg_$i = weight_sum_g[xp * W + yp + $i];

            for (int i=0; i<3; i++){
                output_g[i][xp][yp+$i] /= wg_$i;
            }
        }
    }
    // $end_unroll

    // Final Weight Normalization B
    // $unroll 4
    scalar wb_$i;
    for(int xp = R + F_B; xp < W - R - F_B; ++xp) {
        for(int yp = R + F_B; yp < H - R - F_B; yp+=$n) {
        
            wb_$i = weight_sum_b[xp * W + yp + $i];

            for (int i=0; i<3; i++){
                output_b[i][xp][yp+$i] /= wb_$i;
            }
        }
    }
    // $end_unroll


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

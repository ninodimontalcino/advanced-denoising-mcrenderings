#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all(scalar* sure, scalar* c, scalar* c_var, scalar* cand_r, scalar* cand_g, scalar* cand_b, int W, int H){

    //int WH = W*H;
    
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


void filtering_basic(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int W, int H){
    
    //int WH = W*H;

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
                        scalar sqdist = c[3 * (xp * W + yp) + i] - c[3 * (xq * W + yq) + i];
                        sqdist *= sqdist;
                        scalar var_cancel = c_var[3 * (xp * W + yp) + i] + fmin(c_var[3 * (xp * W + yp) + i], c_var[3 * (xq * W + yq) + i]);
                        scalar normalization = EPSILON + k_c_squared*(c_var[3 * (xp * W + yp) + i] + c_var[3 * (xq * W + yq) + i]);
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


void feature_prefiltering(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int W, int H){

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

void candidate_filtering(scalar* output, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters p, int W, int H){

    //int WH = W * H;

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

    for(int x =  p.r + p.f; x < W - p.r - p.f; ++x) {
        for(int y = p.r + p.f; y < H - p.r - p.f; ++y) {
            for(int i=0; i<NB_FEATURES;++i) {
                scalar diffL = features[3 * (x * W + y) + i] - features[i + 3 * ((x-1) * W + y)];
                scalar diffR = features[3 * (x * W + y) + i] - features[i + 3 * ((x+1) * W + y)];
                scalar diffU = features[3 * (x * W + y) + i] - features[i + 3 * (x * W + y - 1) ];
                scalar diffD = features[3 * (x * W + y) + i] - features[i + 3 * (x * W + y + 1) ];

                gradients[3 * (x * W + y) + i] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
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
                        scalar sqdist = color[3 * (xp * W + yp) + i] - color[3 * (xq * W + yq) + i];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[3 * (xp * W + yp) + i] + fmin(color_var[3 * (xp * W + yp) + i], color_var[3 * (xq * W + yq) + i]);
                        scalar normalization = EPSILON + k_c_squared*(color_var[3 * (xp * W + yp) + i] + color_var[3 * (xq * W + yq) + i]);
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
                        scalar sqdist = features[3 * (xp * W + yp) + j] - features[3 * (xq * W + yq) + j];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[3 * (xp * W + yp) + j] + fmin(features_var[3 * (xp * W + yp) + j], features_var[3 * (xq * W + yq) + j]);
                        scalar normalization = k_f_squared*fmax(p.tau, fmax(features_var[3 * (xp * W + yp) + j], gradients[3 * (xp * W + yp) + j]));
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
                        output[3 * (xp * W + yp) + i] += weight * color[3 * (xq * W + yq) + i];
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
            }
        }
    }

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r + p.f; yp++){
                output[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
            for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
                output[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output[3 * ((W - xp - 1) * W + yp) + i] = color[3 * ((W - xp - 1) * W + yp) + i];
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


void candidate_filtering_all(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int W, int H){

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
    //int WH = W * H;

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

    for(int x =  R+f_min; x < W - R - f_min; ++x) {
        for(int y =  R+f_min; y < H-  R - f_min; ++y) {
            for(int i=0; i<NB_FEATURES;++i) {
                scalar diffL = features[3 * (x * W + y) + i] - features[3 * ((x-1) * W + y) + i];
                scalar diffR = features[3 * (x * W + y) + i] - features[3 * ((x+1) * W + y) + i];
                scalar diffU = features[3 * (x * W + y) + i] - features[3 * (x * W + y - 1) + i];
                scalar diffD = features[3 * (x * W + y) + i] - features[3 * (x * W + y + 1) + i];

                gradients[3 * (x * W + y) + i] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
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
                        scalar sqdist = color[3 * (xp * W + yp) + i] - color[3 * (xq * W + yq) + i];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[3 * (xp * W + yp) + i] + fmin(color_var[3 * (xp * W + yp) + i], color_var[3 * (xq * W + yq) + i]);
                        scalar var_term = color_var[3 * (xp * W + yp) + i] + color_var[3 * (xq * W + yq) + i];
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
                        scalar sqdist = features[3 * (xp * W + yp) + j] - features[3 * (xq * W + yq) + j];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[3 * (xp * W + yp) + j] + fmin(features_var[3 * (xp * W + yp) + j], features_var[3 * (xq * W + yq) + j]);
                        scalar var_max = fmax(features_var[3 * (xp * W + yp) + j], gradients[3 * (xp * W + yp) + j]);
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
                    weight_sum[3 * (xp * W + yp)] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_r[3 * (xp * W + yp) + i] += weight * color[3 * (xq * W + yq) + i];
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
                    weight_sum[1 + 3 * (xp * W + yp)] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_g[3 * (xp * W + yp) + i] += weight * color[3 * (xq * W + yq) + i];
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
                    weight_sum[2 + 3 * (xp * W + yp)] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_b[3 * (xp * W + yp) + i] += weight * color[3 * (xq * W + yq) + i];
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < W - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < H - R - f_r; ++yp) {
        
            scalar w = weight_sum[3 * (xp * W + yp)];
            for (int i=0; i<3; i++){
                output_r[3 * (xp * W + yp) + i] /= w;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < W - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < H - R - f_g; ++yp) {
        
            scalar w = weight_sum[1 + 3 * (xp * W + yp)];
            for (int i=0; i<3; i++){
                output_g[3 * (xp * W + yp) + i] /= w;
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < W - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < H - R - f_b; ++yp) {
        
            scalar w = weight_sum[2 + 3 * (xp * W + yp)];
            for (int i=0; i<3; i++){
                output_b[3 * (xp * W + yp) + i] /= w;
            }
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + f_r; yp++){
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
        for(int xp = 0; xp < R + f_r; xp++){
            for (int yp = R + f_r ; yp < H - R - f_r; yp++){
            
                output_r[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_r[3 * ((W - xp - 1) * W + yp) + i ] = color[3 * ((W - xp - 1) * W + yp) + i ];
                output_b[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_b[3 * ((W - xp - 1) * W + yp) + i ] = color[3 * ((W - xp - 1) * W + yp) + i ];
             }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < R + f_g; yp++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[i + 3 * (xp * W + H - yp - 1)] = color[i + 3 * (xp * W + H - yp - 1)];
            }
        }
        for(int xp = 0; xp < R + f_g; xp++){
            for (int yp = R + f_g ; yp < H - R - f_g; yp++){
                output_g[3 * (xp * W + yp) + i] = color[3 * (xp * W + yp) + i];
                output_g[3 * ((W - xp - 1) * W + yp) + i ] = color[3 * ((W - xp - 1) * W + yp) + i ];
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
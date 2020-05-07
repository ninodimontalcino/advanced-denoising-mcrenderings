#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"


void sure_all(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int img_width, int img_height){
    
    scalar d_r, d_g, d_b, v;
    
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){

            scalar sure_r = 0.f;
            scalar sure_g = 0.f;
            scalar sure_b = 0.f;

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Calculate terms
                d_r = cand_r[i][x][y] - c[i][x][y];
                d_r *= d_r;
                d_g = cand_g[i][x][y] - c[i][x][y];
                d_g *= d_g;
                d_b = cand_b[i][x][y] - c[i][x][y];
                d_b *= d_b;
                v = c_var[i][x][y];
                v *= v;

                // Summing up
                sure_r += d_r - v; 
                sure_g += d_g - v; 
                sure_b += d_b - v; 

            }
            // Store sure error estimate
            sure[0][x][y] = sure_r;
            sure[1][x][y] = sure_g;
            sure[2][x][y] = sure_b;
        }
    }
}


void filtering_basic(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int img_width, int img_height){
    

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    channel weight_sum;
    allocate_channel_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp, temp2;
    allocate_channel(&temp, img_width, img_height); 
    allocate_channel(&temp2, img_width, img_height); 

    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = p.r; xp < img_width - p.r; ++xp) {
                for(int yp = p.r; yp < img_height - p.r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance = 0;
                    for (int i=0; i<3; i++){                        
                        scalar sqdist = c[i][xp][yp] - c[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = c_var[i][xp][yp] + fmin(c_var[i][xp][yp], c_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(c_var[i][xp][yp] + c_var[i][xq][yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp][yp] = distance;

                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < img_width - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp[xp][yp+k];
                    }
                    temp2[xp][yp] = sum;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[xp+k][yp];
                    }
                    scalar weight = exp(-fmax(0.f, (sum / neigh)));
                    weight_sum[xp][yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * input[i][xq][yq];
                    }
                }
            }

        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f ; yp < img_height - p.r - p.f; ++yp) {     
            scalar w = weight_sum[xp][yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
             for (int yp = p.r+p.f ; yp < img_height - p.r - p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
            }
        }

    }

    free_channel(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2, img_width); 
}


void feature_prefiltering(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int img_width, int img_height){

    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;

    // Allocate buffer weights_sum for normalizing
    channel weight_sum;
    allocate_channel_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp, temp2;
    allocate_channel(&temp, img_width, img_height); 
    allocate_channel(&temp2, img_width, img_height); 

    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);

    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = p.r; xp < img_width - p.r; ++xp) {
                for(int yp = p.r; yp < img_height - p.r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance = 0;
                    for (int i=0; i<3; i++){                        
                        scalar sqdist = features[i][xp][yp] - features[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[i][xp][yp] + fmin(features_var[i][xp][yp], features_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(features_var[i][xp][yp] + features_var[i][xq][yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp][yp] = distance;

                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < img_width - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {
                    
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp[xp][yp+k];
                    }
                    temp2[xp][yp] = sum;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[xp+k][yp];
                    }
                    scalar weight = exp(-fmax(0.f, (sum / neigh)));
                    weight_sum[xp][yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * features[i][xq][yq];
                        output_var[i][xp][yp] += weight * features_var[i][xq][yq];
                    }
                }
            }


        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {
        
            scalar w = weight_sum[xp][yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
                output_var[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = features[i][xp][yp];
                output[i][xp][img_height - yp - 1] = features[i][xp][img_height - yp - 1];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][xp][img_height - yp - 1] = features_var[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
             for (int yp = p.r+p.f ; yp < img_height - p.r - p.f; yp++){
                output[i][xp][yp] = features[i][xp][yp];
                output[i][img_width - xp - 1][yp] = features[i][img_width - xp - 1][yp];
                output_var[i][xp][yp] = features_var[i][xp][yp];
                output_var[i][xp][img_height - yp - 1] = features_var[i][xp][img_height - yp - 1];
            }
        }

    }

    // Free memory
    free_channel(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2, img_width); 

}

void candidate_filtering(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int img_width, int img_height){


    // Handling Inner Part   
    // -------------------
    scalar k_c_squared = p.kc * p.kc;
    scalar k_f_squared = p.kf * p.kf;

    // Allocate buffer weights_sum for normalizing
    channel weight_sum;
    allocate_channel_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp, temp2;
    allocate_channel(&temp, img_width, img_height); 
    allocate_channel(&temp2, img_width, img_height); 

    // Init feature weights channel
    channel feature_weights;
    allocate_channel(&feature_weights, img_width, img_height);

    // Compute gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for(int i=0; i<NB_FEATURES;++i) {
        compute_gradient(gradients[i], features[i], p.r+p.f, img_width, img_height);
    }

    // Precompute size of neighbourhood
    scalar neigh = 3*(2*p.f+1)*(2*p.f+1);


    // Covering the neighbourhood
    for (int r_x = -p.r; r_x <= p.r; r_x++){
        for (int r_y = -p.r; r_y <= p.r; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = p.r; xp < img_width - p.r; ++xp) {
                for(int yp = p.r; yp < img_height - p.r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance = 0;
                    for (int i=0; i<3; i++){                        
                        scalar sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        scalar normalization = EPSILON + k_c_squared*(color_var[i][xp][yp] + color_var[i][xq][yq]);
                        distance += (sqdist - var_cancel) / normalization;
                    }

                    temp[xp][yp] = distance;

                }
            }

            // Compute features
            for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute feature weight
                    scalar df = 0.f;
                    for(int j=0; j<NB_FEATURES;++j){
                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        scalar normalization = k_f_squared*fmax(p.tau, fmax(features_var[j][xp][yp], gradients[j][xp][yp]));
                        df = fmax(df, (sqdist - var_cancel)/normalization);
                    }
                    feature_weights[xp][yp] = exp(-df);
                }
            }

            // Apply Box-Filtering for Patch Contribution => Use Box-Filter Seperability
            // (1) Convolve along height
            for(int xp = p.r; xp < img_width - p.r; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp[xp][yp+k];
                    }
                    temp2[xp][yp] = sum;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
                for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k=-p.f; k<=p.f; k++){
                        sum += temp2[xp+k][yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh)));
                    
                    scalar weight = fmin(color_weight, feature_weights[xp][yp]);
                    weight_sum[xp][yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }
        }
    }

    // Final Weight Normalization
    for(int xp = p.r + p.f; xp < img_width - p.r - p.f; ++xp) {
        for(int yp = p.r + p.f; yp < img_height - p.r - p.f; ++yp) {
        
            scalar w = weight_sum[xp][yp];
            for (int i=0; i<3; i++){
                output[i][xp][yp] /= w;
            }
        }
    }

    // Handle Border Cases
    // ---------------------
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r + p.f; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < p.r+p.f; xp++){
            for (int yp = p.r+p.f ; yp < img_height - p.r - p.f; yp++){
                output[i][xp][yp] = color[i][xp][yp];
                output[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
            }
        }
    }

    // Free memory
    free_channel(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2, img_width);
    free_channel(&feature_weights, img_width);
    free_buffer(&gradients, img_width);
}



// ====================================================================================================================================================================================================================================
// !!! TO BE IMPROVED => still more precomputations possible
// ====================================================================================================================================================================================================================================

void candidate_filtering_all(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int img_width, int img_height){

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


    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(f_r, fmax(f_g, f_b));
    int f_min = fmin(f_r, fmin(f_g, f_b));
    int R = p[0].r;


    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    buffer weight_sum;
    allocate_buffer_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp;
    channel temp2_r;
    channel temp2_g;
    allocate_channel(&temp, img_width, img_height); 
    allocate_channel(&temp2_r, img_width, img_height); 
    allocate_channel(&temp2_g, img_width, img_height); 

    // Allocate feature weights buffer
    channel features_weights_r;
    channel features_weights_b;
    allocate_channel(&features_weights_r, img_width, img_height);
    allocate_channel(&features_weights_b, img_width, img_height);


    // Compute gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for(int i=0; i<NB_FEATURES;++i) {
        compute_gradient(gradients[i], features[i], R+f_min, img_width, img_height);
    }

    // Precompute size of neighbourhood
    scalar neigh_r = 3*(2*f_r+1)*(2*f_r+1);
    scalar neigh_g = 3*(2*f_g+1)*(2*f_g+1);
    scalar neigh_b = 3*(2*f_b+1)*(2*f_b+1);

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = R; xp < img_width - R; ++xp) {
                for(int yp = R; yp < img_height - R; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance_r = 0.f;

                    for (int i=0; i<3; i++){                        
                        scalar sqdist = color[i][xp][yp] - color[i][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        scalar var_term = color_var[i][xp][yp] + color_var[i][xq][yq];
                        scalar normalization_r = EPSILON + k_c_squared_r*(var_term);
                        scalar dist_var = sqdist - var_cancel;
                        distance_r += (dist_var) / normalization_r;
                    }

                    temp[xp][yp] = distance_r;

                }
            }

            // Precompute feature weights
            for(int xp = R + f_min; xp < img_width - R - f_min; ++xp) {
                for(int yp = R + f_min; yp < img_height - R - f_min; ++yp) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r = 0.f;
                    scalar df_b = 0.f;

                    for(int j=0; j<NB_FEATURES;++j){
                        scalar sqdist = features[j][xp][yp] - features[j][xq][yq];
                        sqdist *= sqdist;
                        scalar var_cancel = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        scalar var_max = fmax(features_var[j][xp][yp], gradients[j][xp][yp]);
                        scalar normalization_r = k_f_squared_r*fmax(tau_r, var_max);
                        scalar normalization_b = k_f_squared_b*fmax(tau_b, var_max);
                        scalar dist_var = sqdist - var_cancel;
                        df_r = fmax(df_r, (dist_var)/normalization_r);
                        df_b = fmax(df_b, (dist_var)/normalization_b);
                    }

                    features_weights_r[xp][yp] = exp(-df_r);
                    features_weights_b[xp][yp] = exp(-df_b);
                } 
            }
            

            // Next Steps: Box-Filtering for Patch Contribution 
            // => Use Box-Filter Seperability => linear scans of data
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < img_width - R; ++xp) {
                for(int yp = R + f_r; yp < img_height - R - f_r; ++yp) {
                    scalar sum_r = 0.f;
                    for (int k=-f_r; k<=f_r; k++){
                        sum_r += temp[xp][yp+k];
                    }
                    temp2_r[xp][yp] = sum_r;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_r; xp < img_width - R - f_r; ++xp) {
                for(int yp = R + f_r; yp < img_height - R - f_r; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k=-f_r; k<=f_r; k++){
                        sum += temp2_r[xp+k][yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh_r)));

                    // Compute final weight
                    scalar weight = fmin(color_weight, features_weights_r[xp][yp]);
                    weight_sum[0][xp][yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < img_width - R; ++xp) {
                for(int yp = R + f_g; yp < img_height - R - f_g; ++yp) {
                    scalar sum_g = 0.f;
                    for (int k=-f_g; k<=f_g; k++){
                        sum_g += temp[xp][yp+k];
                    }
                    temp2_g[xp][yp] = sum_g;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_g; xp < img_width - R - f_g; ++xp) {
                for(int yp = R + f_g; yp < img_height - R - f_g; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum = 0.f;
                    for (int k=-f_g; k<=f_g; k++){
                        sum += temp2_g[xp+k][yp];
                    }
                    scalar color_weight = exp(-fmax(0.f, (sum / neigh_g)));
                    
                    // Compute final weight
                    scalar weight = fmin(color_weight, features_weights_r[xp][yp]);
                    weight_sum[1][xp][yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = R + f_b; xp < img_width - R - f_b; ++xp) {
                for(int yp = R + f_b; yp < img_height - R - f_b; ++yp) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight = features_weights_b[xp][yp];
                    weight_sum[2][xp][yp] += weight;
                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp] += weight * color[i][xq][yq];
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < img_width - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < img_height - R - f_r; ++yp) {
        
            scalar w = weight_sum[0][xp][yp];
            for (int i=0; i<3; i++){
                output_r[i][xp][yp] /= w;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < img_width - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < img_height - R - f_g; ++yp) {
        
            scalar w = weight_sum[1][xp][yp];
            for (int i=0; i<3; i++){
                output_g[i][xp][yp] /= w;
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < img_width - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < img_height - R - f_b; ++yp) {
        
            scalar w = weight_sum[2][xp][yp];
            for (int i=0; i<3; i++){
                output_b[i][xp][yp] /= w;
            }
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < R + f_r; yp++){
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_r; xp++){
            for (int yp = R + f_r ; yp < img_height - R - f_r; yp++){
            
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
             }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < R + f_g; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_g; xp++){
            for (int yp = R + f_g ; yp < img_height - R - f_g; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
            }
        }
    }

    // Free memory
    free_buffer(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2_r, img_width);
    free_channel(&temp2_g, img_width);
    free_channel(&features_weights_r, img_width);
    free_channel(&features_weights_b, img_width);
    free_buffer(&gradients, img_width);

}



// =====================================================================================================================
// SOME TESTS
// =====================================================================================================================


void candidate_filtering_all2(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int img_width, int img_height){

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


    // Determinte max f => R is fixed to the same for all
    int f_max = fmax(f_r, fmax(f_g, f_b));
    int f_min = fmin(f_r, fmin(f_g, f_b));
    int R = p[0].r;


    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    buffer weight_sum;
    allocate_buffer_zero(&weight_sum, img_width, img_height);

    // Init temp channel
    channel temp;
    channel temp2_r;
    channel temp2_g;
    allocate_channel(&temp, img_width, img_height); 
    allocate_channel(&temp2_r, img_width, img_height); 
    allocate_channel(&temp2_g, img_width, img_height); 

    // Allocate feature weights buffer
    channel features_weights_r;
    channel features_weights_b;
    allocate_channel(&features_weights_r, img_width, img_height);
    allocate_channel(&features_weights_b, img_width, img_height);


    // Compute gradients
    buffer gradients;
    allocate_buffer(&gradients, img_width, img_height);
    for(int i=0; i<NB_FEATURES;++i) {
        compute_gradient(gradients[i], features[i], R+f_min, img_width, img_height);
    }

    // Precompute size of neighbourhood
    scalar neigh_r = 3*(2*f_r+1)*(2*f_r+1);
    scalar neigh_g = 3*(2*f_g+1)*(2*f_g+1);
    scalar neigh_b = 3*(2*f_b+1)*(2*f_b+1);

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // Compute Color Weight for all pixels with fixed r
            for(int xp = R; xp < img_width - R; ++xp) {
                for(int yp = R; yp < img_height - R; yp+=16) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar distance_r_0 = 0.f;
                    scalar distance_r_1 = 0.f;
                    scalar distance_r_2 = 0.f;
                    scalar distance_r_3 = 0.f;
                    scalar distance_r_4 = 0.f;
                    scalar distance_r_5 = 0.f;
                    scalar distance_r_6 = 0.f;
                    scalar distance_r_7 = 0.f;
                    scalar distance_r_8 = 0.f;
                    scalar distance_r_9 = 0.f;
                    scalar distance_r_10 = 0.f;
                    scalar distance_r_11 = 0.f;
                    scalar distance_r_12 = 0.f;
                    scalar distance_r_13 = 0.f;
                    scalar distance_r_14 = 0.f;
                    scalar distance_r_15 = 0.f;

                    for (int i=0; i<3; i++){   

                        scalar sqdist_0 = color[i][xp][yp] - color[i][xq][yq];
                        scalar sqdist_1 = color[i][xp][yp+1] - color[i][xq][yq+1];
                        scalar sqdist_2 = color[i][xp][yp+2] - color[i][xq][yq+2];
                        scalar sqdist_3 = color[i][xp][yp+3] - color[i][xq][yq+3];
                        scalar sqdist_4 = color[i][xp][yp+4] - color[i][xq][yq+4];
                        scalar sqdist_5 = color[i][xp][yp+5] - color[i][xq][yq+5];
                        scalar sqdist_6 = color[i][xp][yp+6] - color[i][xq][yq+6];
                        scalar sqdist_7 = color[i][xp][yp+7] - color[i][xq][yq+7];
                        scalar sqdist_8 = color[i][xp][yp+8] - color[i][xq][yq+8];
                        scalar sqdist_9 = color[i][xp][yp+9] - color[i][xq][yq+9];
                        scalar sqdist_10 = color[i][xp][yp+10] - color[i][xq][yq+10];
                        scalar sqdist_11 = color[i][xp][yp+11] - color[i][xq][yq+11];
                        scalar sqdist_12 = color[i][xp][yp+12] - color[i][xq][yq+12];
                        scalar sqdist_13 = color[i][xp][yp+13] - color[i][xq][yq+13];
                        scalar sqdist_14 = color[i][xp][yp+14] - color[i][xq][yq+14];
                        scalar sqdist_15 = color[i][xp][yp+15] - color[i][xq][yq+15];

                        sqdist_0 *= sqdist_0;
                        sqdist_1 *= sqdist_1;
                        sqdist_2 *= sqdist_2;
                        sqdist_3 *= sqdist_3;
                        sqdist_4 *= sqdist_4;
                        sqdist_5 *= sqdist_5;
                        sqdist_6 *= sqdist_6;
                        sqdist_7 *= sqdist_7;
                        sqdist_8 *= sqdist_8;
                        sqdist_9 *= sqdist_9;
                        sqdist_10 *= sqdist_10;
                        sqdist_11 *= sqdist_11;
                        sqdist_12 *= sqdist_12;
                        sqdist_13 *= sqdist_13;
                        sqdist_14 *= sqdist_14;
                        sqdist_15 *= sqdist_15;

                        scalar var_cancel_0 = color_var[i][xp][yp] + fmin(color_var[i][xp][yp], color_var[i][xq][yq]);
                        scalar var_cancel_1 = color_var[i][xp][yp+1] + fmin(color_var[i][xp][yp+1], color_var[i][xq][yq+1]);
                        scalar var_cancel_2 = color_var[i][xp][yp+2] + fmin(color_var[i][xp][yp+2], color_var[i][xq][yq+2]);
                        scalar var_cancel_3 = color_var[i][xp][yp+3] + fmin(color_var[i][xp][yp+3], color_var[i][xq][yq+3]);
                        scalar var_cancel_4 = color_var[i][xp][yp+4] + fmin(color_var[i][xp][yp+4], color_var[i][xq][yq+4]);
                        scalar var_cancel_5 = color_var[i][xp][yp+5] + fmin(color_var[i][xp][yp+5], color_var[i][xq][yq+5]);
                        scalar var_cancel_6 = color_var[i][xp][yp+6] + fmin(color_var[i][xp][yp+6], color_var[i][xq][yq+6]);
                        scalar var_cancel_7 = color_var[i][xp][yp+7] + fmin(color_var[i][xp][yp+7], color_var[i][xq][yq+7]);
                        scalar var_cancel_8 = color_var[i][xp][yp+8] + fmin(color_var[i][xp][yp+8], color_var[i][xq][yq+8]);
                        scalar var_cancel_9 = color_var[i][xp][yp+9] + fmin(color_var[i][xp][yp+9], color_var[i][xq][yq+9]);
                        scalar var_cancel_10 = color_var[i][xp][yp+10] + fmin(color_var[i][xp][yp+10], color_var[i][xq][yq+10]);
                        scalar var_cancel_11 = color_var[i][xp][yp+11] + fmin(color_var[i][xp][yp+11], color_var[i][xq][yq+11]);
                        scalar var_cancel_12 = color_var[i][xp][yp+12] + fmin(color_var[i][xp][yp+12], color_var[i][xq][yq+12]);
                        scalar var_cancel_13 = color_var[i][xp][yp+13] + fmin(color_var[i][xp][yp+13], color_var[i][xq][yq+13]);
                        scalar var_cancel_14 = color_var[i][xp][yp+14] + fmin(color_var[i][xp][yp+14], color_var[i][xq][yq+14]);
                        scalar var_cancel_15 = color_var[i][xp][yp+15] + fmin(color_var[i][xp][yp+15], color_var[i][xq][yq+15]);

                        scalar var_term_0 = color_var[i][xp][yp] + color_var[i][xq][yq];
                        scalar var_term_1 = color_var[i][xp][yp+1] + color_var[i][xq][yq+1];
                        scalar var_term_2 = color_var[i][xp][yp+2] + color_var[i][xq][yq+2];
                        scalar var_term_3 = color_var[i][xp][yp+3] + color_var[i][xq][yq+3];
                        scalar var_term_4 = color_var[i][xp][yp+4] + color_var[i][xq][yq+4];
                        scalar var_term_5 = color_var[i][xp][yp+5] + color_var[i][xq][yq+5];
                        scalar var_term_6 = color_var[i][xp][yp+6] + color_var[i][xq][yq+6];
                        scalar var_term_7 = color_var[i][xp][yp+7] + color_var[i][xq][yq+7];
                        scalar var_term_8 = color_var[i][xp][yp+8] + color_var[i][xq][yq+8];
                        scalar var_term_9 = color_var[i][xp][yp+9] + color_var[i][xq][yq+9];
                        scalar var_term_10 = color_var[i][xp][yp+10] + color_var[i][xq][yq+10];
                        scalar var_term_11 = color_var[i][xp][yp+11] + color_var[i][xq][yq+11];
                        scalar var_term_12 = color_var[i][xp][yp+12] + color_var[i][xq][yq+12];
                        scalar var_term_13 = color_var[i][xp][yp+13] + color_var[i][xq][yq+13];
                        scalar var_term_14 = color_var[i][xp][yp+14] + color_var[i][xq][yq+14];
                        scalar var_term_15 = color_var[i][xp][yp+15] + color_var[i][xq][yq+15];

                        scalar normalization_r_0 = EPSILON + k_c_squared_r*(var_term_0);
                        scalar normalization_r_1 = EPSILON + k_c_squared_r*(var_term_1);
                        scalar normalization_r_2 = EPSILON + k_c_squared_r*(var_term_2);
                        scalar normalization_r_3 = EPSILON + k_c_squared_r*(var_term_3);
                        scalar normalization_r_4 = EPSILON + k_c_squared_r*(var_term_4);
                        scalar normalization_r_5 = EPSILON + k_c_squared_r*(var_term_5);
                        scalar normalization_r_6 = EPSILON + k_c_squared_r*(var_term_6);
                        scalar normalization_r_7 = EPSILON + k_c_squared_r*(var_term_7);
                        scalar normalization_r_8 = EPSILON + k_c_squared_r*(var_term_8);
                        scalar normalization_r_9 = EPSILON + k_c_squared_r*(var_term_9);
                        scalar normalization_r_10 = EPSILON + k_c_squared_r*(var_term_10);
                        scalar normalization_r_11 = EPSILON + k_c_squared_r*(var_term_11);
                        scalar normalization_r_12 = EPSILON + k_c_squared_r*(var_term_12);
                        scalar normalization_r_13 = EPSILON + k_c_squared_r*(var_term_13);
                        scalar normalization_r_14 = EPSILON + k_c_squared_r*(var_term_14);
                        scalar normalization_r_15 = EPSILON + k_c_squared_r*(var_term_15);

                        scalar dist_var_0 = sqdist_0 - var_cancel_0;
                        scalar dist_var_1 = sqdist_1 - var_cancel_1;
                        scalar dist_var_2 = sqdist_2 - var_cancel_2;
                        scalar dist_var_3 = sqdist_3 - var_cancel_3;
                        scalar dist_var_4 = sqdist_4 - var_cancel_4;
                        scalar dist_var_5 = sqdist_5 - var_cancel_5;
                        scalar dist_var_6 = sqdist_6 - var_cancel_6;
                        scalar dist_var_7 = sqdist_7 - var_cancel_7;
                        scalar dist_var_8 = sqdist_8 - var_cancel_8;
                        scalar dist_var_9 = sqdist_9 - var_cancel_9;
                        scalar dist_var_10 = sqdist_10 - var_cancel_10;
                        scalar dist_var_11 = sqdist_11 - var_cancel_11;
                        scalar dist_var_12 = sqdist_12 - var_cancel_12;
                        scalar dist_var_13 = sqdist_13 - var_cancel_13;
                        scalar dist_var_14 = sqdist_14 - var_cancel_14;
                        scalar dist_var_15 = sqdist_15 - var_cancel_15;

                        distance_r_0 += (dist_var_0) / normalization_r_0;
                        distance_r_1 += (dist_var_1) / normalization_r_1;
                        distance_r_2 += (dist_var_2) / normalization_r_2;
                        distance_r_3 += (dist_var_3) / normalization_r_3;
                        distance_r_4 += (dist_var_4) / normalization_r_4;
                        distance_r_5 += (dist_var_5) / normalization_r_5;
                        distance_r_6 += (dist_var_6) / normalization_r_6;
                        distance_r_7 += (dist_var_7) / normalization_r_7;
                        distance_r_8 += (dist_var_8) / normalization_r_8;
                        distance_r_9 += (dist_var_9) / normalization_r_9;
                        distance_r_10 += (dist_var_10) / normalization_r_10;
                        distance_r_11 += (dist_var_11) / normalization_r_11;
                        distance_r_12 += (dist_var_12) / normalization_r_12;
                        distance_r_13 += (dist_var_13) / normalization_r_13;
                        distance_r_14 += (dist_var_14) / normalization_r_14;
                        distance_r_15 += (dist_var_15) / normalization_r_15;
                    }

                    temp[xp][yp] = distance_r_0;
                    temp[xp][yp+1] = distance_r_1;
                    temp[xp][yp+2] = distance_r_2;
                    temp[xp][yp+3] = distance_r_3;
                    temp[xp][yp+4] = distance_r_4;
                    temp[xp][yp+5] = distance_r_5;
                    temp[xp][yp+6] = distance_r_6;
                    temp[xp][yp+7] = distance_r_7;
                    temp[xp][yp+8] = distance_r_8;
                    temp[xp][yp+9] = distance_r_9;
                    temp[xp][yp+10] = distance_r_10;
                    temp[xp][yp+11] = distance_r_11;
                    temp[xp][yp+12] = distance_r_12;
                    temp[xp][yp+13] = distance_r_13;
                    temp[xp][yp+14] = distance_r_14;
                    temp[xp][yp+15] = distance_r_15;


                }
            }

            // Precompute feature weights
            for(int xp = R + f_min; xp < img_width - R - f_min; ++xp) {
                for(int yp = R + f_min; yp < img_height - R - f_min; yp+=8) {
                    
                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar df_r_0 = 0.f;
                    scalar df_r_1 = 0.f;
                    scalar df_r_2 = 0.f;
                    scalar df_r_3 = 0.f;
                    scalar df_r_4 = 0.f;
                    scalar df_r_5 = 0.f;
                    scalar df_r_6 = 0.f;
                    scalar df_r_7 = 0.f;

                    scalar df_b_0 = 0.f;
                    scalar df_b_1 = 0.f;
                    scalar df_b_2 = 0.f;
                    scalar df_b_3 = 0.f;
                    scalar df_b_4 = 0.f;
                    scalar df_b_5 = 0.f;
                    scalar df_b_6 = 0.f;
                    scalar df_b_7 = 0.f;

                    for(int j=0; j<NB_FEATURES;++j){
                        
                        scalar sqdist_0 = features[j][xp][yp] - features[j][xq][yq];
                        scalar sqdist_1 = features[j][xp][yp+1] - features[j][xq][yq+1];
                        scalar sqdist_2 = features[j][xp][yp+2] - features[j][xq][yq+2];
                        scalar sqdist_3 = features[j][xp][yp+3] - features[j][xq][yq+3];
                        scalar sqdist_4 = features[j][xp][yp+4] - features[j][xq][yq+4];
                        scalar sqdist_5 = features[j][xp][yp+5] - features[j][xq][yq+5];
                        scalar sqdist_6 = features[j][xp][yp+6] - features[j][xq][yq+6];
                        scalar sqdist_7 = features[j][xp][yp+7] - features[j][xq][yq+7];

                        sqdist_0 *= sqdist_0;
                        sqdist_1 *= sqdist_1;
                        sqdist_2 *= sqdist_2;
                        sqdist_3 *= sqdist_3;
                        sqdist_4 *= sqdist_4;
                        sqdist_5 *= sqdist_5;
                        sqdist_6 *= sqdist_6;
                        sqdist_7 *= sqdist_7;

                        scalar var_cancel_0 = features_var[j][xp][yp] + fmin(features_var[j][xp][yp], features_var[j][xq][yq]);
                        scalar var_cancel_1 = features_var[j][xp][yp+1] + fmin(features_var[j][xp][yp+1], features_var[j][xq][yq+1]);
                        scalar var_cancel_2 = features_var[j][xp][yp+2] + fmin(features_var[j][xp][yp+2], features_var[j][xq][yq+2]);
                        scalar var_cancel_3 = features_var[j][xp][yp+3] + fmin(features_var[j][xp][yp+3], features_var[j][xq][yq+3]);
                        scalar var_cancel_4 = features_var[j][xp][yp+4] + fmin(features_var[j][xp][yp+4], features_var[j][xq][yq+4]);
                        scalar var_cancel_5 = features_var[j][xp][yp+5] + fmin(features_var[j][xp][yp+5], features_var[j][xq][yq+5]);
                        scalar var_cancel_6 = features_var[j][xp][yp+6] + fmin(features_var[j][xp][yp+6], features_var[j][xq][yq+6]);
                        scalar var_cancel_7 = features_var[j][xp][yp+7] + fmin(features_var[j][xp][yp+7], features_var[j][xq][yq+7]);
                        
                        scalar var_max_0 = fmax(features_var[j][xp][yp], gradients[j][xp][yp]);
                        scalar var_max_1 = fmax(features_var[j][xp][yp+1], gradients[j][xp][yp+1]);
                        scalar var_max_2 = fmax(features_var[j][xp][yp+2], gradients[j][xp][yp+2]);
                        scalar var_max_3 = fmax(features_var[j][xp][yp+3], gradients[j][xp][yp+3]);
                        scalar var_max_4 = fmax(features_var[j][xp][yp+4], gradients[j][xp][yp+4]);
                        scalar var_max_5 = fmax(features_var[j][xp][yp+5], gradients[j][xp][yp+5]);
                        scalar var_max_6 = fmax(features_var[j][xp][yp+6], gradients[j][xp][yp+6]);
                        scalar var_max_7 = fmax(features_var[j][xp][yp+7], gradients[j][xp][yp+7]);

                        scalar normalization_r_0 = k_f_squared_r*fmax(tau_r, var_max_0);
                        scalar normalization_r_1 = k_f_squared_r*fmax(tau_r, var_max_1);
                        scalar normalization_r_2 = k_f_squared_r*fmax(tau_r, var_max_2);
                        scalar normalization_r_3 = k_f_squared_r*fmax(tau_r, var_max_3);
                        scalar normalization_r_4 = k_f_squared_r*fmax(tau_r, var_max_4);
                        scalar normalization_r_5 = k_f_squared_r*fmax(tau_r, var_max_5);
                        scalar normalization_r_6 = k_f_squared_r*fmax(tau_r, var_max_6);
                        scalar normalization_r_7 = k_f_squared_r*fmax(tau_r, var_max_7);

                        scalar normalization_b_0 = k_f_squared_b*fmax(tau_b, var_max_0);
                        scalar normalization_b_1 = k_f_squared_b*fmax(tau_b, var_max_1);
                        scalar normalization_b_2 = k_f_squared_b*fmax(tau_b, var_max_2);
                        scalar normalization_b_3 = k_f_squared_b*fmax(tau_b, var_max_3);
                        scalar normalization_b_4 = k_f_squared_b*fmax(tau_b, var_max_4);
                        scalar normalization_b_5 = k_f_squared_b*fmax(tau_b, var_max_5);
                        scalar normalization_b_6 = k_f_squared_b*fmax(tau_b, var_max_6);
                        scalar normalization_b_7 = k_f_squared_b*fmax(tau_b, var_max_7);

                        scalar dist_var_0 = sqdist_0 - var_cancel_0;
                        scalar dist_var_1 = sqdist_1 - var_cancel_1;
                        scalar dist_var_2 = sqdist_2 - var_cancel_2;
                        scalar dist_var_3 = sqdist_3 - var_cancel_3;
                        scalar dist_var_4 = sqdist_4 - var_cancel_4;
                        scalar dist_var_5 = sqdist_5 - var_cancel_5;
                        scalar dist_var_6 = sqdist_6 - var_cancel_6;
                        scalar dist_var_7 = sqdist_7 - var_cancel_7;

                        df_r_0 = fmax(df_r_0, (dist_var_0)/normalization_r_0);
                        df_r_1 = fmax(df_r_1, (dist_var_1)/normalization_r_1);
                        df_r_2 = fmax(df_r_2, (dist_var_2)/normalization_r_2);
                        df_r_3 = fmax(df_r_3, (dist_var_3)/normalization_r_3);
                        df_r_4 = fmax(df_r_4, (dist_var_4)/normalization_r_4);
                        df_r_5 = fmax(df_r_5, (dist_var_5)/normalization_r_5);
                        df_r_6 = fmax(df_r_6, (dist_var_6)/normalization_r_6);
                        df_r_7 = fmax(df_r_7, (dist_var_7)/normalization_r_7);

                        df_b_0 = fmax(df_b_0, (dist_var_0)/normalization_b_0);
                        df_b_1 = fmax(df_b_1, (dist_var_1)/normalization_b_1);
                        df_b_2 = fmax(df_b_2, (dist_var_2)/normalization_b_2);
                        df_b_3 = fmax(df_b_3, (dist_var_3)/normalization_b_3);
                        df_b_4 = fmax(df_b_4, (dist_var_4)/normalization_b_4);
                        df_b_5 = fmax(df_b_5, (dist_var_5)/normalization_b_5);
                        df_b_6 = fmax(df_b_6, (dist_var_6)/normalization_b_6);
                        df_b_7 = fmax(df_b_7, (dist_var_7)/normalization_b_7);
                    }

                    features_weights_r[xp][yp] = exp(-df_r_0);
                    features_weights_b[xp][yp] = exp(-df_b_0);
                    features_weights_r[xp][yp+1] = exp(-df_r_1);
                    features_weights_b[xp][yp+1] = exp(-df_b_1);
                    features_weights_r[xp][yp+2] = exp(-df_r_2);
                    features_weights_b[xp][yp+2] = exp(-df_b_2);
                    features_weights_r[xp][yp+3] = exp(-df_r_3);
                    features_weights_b[xp][yp+3] = exp(-df_b_3);
                    features_weights_r[xp][yp+4] = exp(-df_r_4);
                    features_weights_b[xp][yp+4] = exp(-df_b_4);
                    features_weights_r[xp][yp+5] = exp(-df_r_5);
                    features_weights_b[xp][yp+5] = exp(-df_b_5);
                    features_weights_r[xp][yp+6] = exp(-df_r_6);
                    features_weights_b[xp][yp+6] = exp(-df_b_6);
                    features_weights_r[xp][yp+7] = exp(-df_r_7);
                    features_weights_b[xp][yp+7] = exp(-df_b_7);
                } 
            }
            

            // Next Steps: Box-Filtering for Patch Contribution 
            // => Use Box-Filter Seperability => linear scans of data
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < img_width - R; ++xp) {
                for(int yp = R + f_r; yp < img_height - R - f_r; yp+=16) {
                    scalar sum_r0 = 0.f;
                    scalar sum_r1 = 0.f;
                    scalar sum_r2 = 0.f;
                    scalar sum_r3 = 0.f;
                    scalar sum_r4 = 0.f;
                    scalar sum_r5 = 0.f;
                    scalar sum_r6 = 0.f;
                    scalar sum_r7 = 0.f;
                    scalar sum_r8 = 0.f;
                    scalar sum_r9 = 0.f;
                    scalar sum_r10 = 0.f;
                    scalar sum_r11 = 0.f;
                    scalar sum_r12 = 0.f;
                    scalar sum_r13 = 0.f;
                    scalar sum_r14 = 0.f;
                    scalar sum_r15 = 0.f;

                    for (int k=-f_r; k<=f_r; k++){
                        sum_r0 += temp[xp][yp+k];
                        sum_r1 += temp[xp][yp+k+1];
                        sum_r2 += temp[xp][yp+k+2];
                        sum_r3 += temp[xp][yp+k+3];
                        sum_r4 += temp[xp][yp+k+4];
                        sum_r5 += temp[xp][yp+k+5];
                        sum_r6 += temp[xp][yp+k+6];
                        sum_r7 += temp[xp][yp+k+7];
                        sum_r8 += temp[xp][yp+k+8];
                        sum_r9 += temp[xp][yp+k+9];
                        sum_r10 += temp[xp][yp+k+10];
                        sum_r11 += temp[xp][yp+k+11];
                        sum_r12 += temp[xp][yp+k+12];
                        sum_r13 += temp[xp][yp+k+13];
                        sum_r14 += temp[xp][yp+k+14];
                        sum_r15 += temp[xp][yp+k+15];
                    }
                    temp2_r[xp][yp] = sum_r0;
                    temp2_r[xp][yp+1] = sum_r1;
                    temp2_r[xp][yp+2] = sum_r2;
                    temp2_r[xp][yp+3] = sum_r3;
                    temp2_r[xp][yp+4] = sum_r4;
                    temp2_r[xp][yp+5] = sum_r5;
                    temp2_r[xp][yp+6] = sum_r6;
                    temp2_r[xp][yp+7] = sum_r7;
                    temp2_r[xp][yp+8] = sum_r8;
                    temp2_r[xp][yp+9] = sum_r9;
                    temp2_r[xp][yp+10] = sum_r10;
                    temp2_r[xp][yp+11] = sum_r11;
                    temp2_r[xp][yp+12] = sum_r12;
                    temp2_r[xp][yp+13] = sum_r13;
                    temp2_r[xp][yp+14] = sum_r14;
                    temp2_r[xp][yp+15] = sum_r15;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_r; xp < img_width - R - f_r; ++xp) {
                for(int yp = R + f_r; yp < img_height - R - f_r; yp+=16) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
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

                    for (int k=-f_r; k<=f_r; k++){
                        sum_0 += temp2_r[xp+k][yp];
                        sum_1 += temp2_r[xp+k][yp+1];
                        sum_2 += temp2_r[xp+k][yp+2];
                        sum_3 += temp2_r[xp+k][yp+3];
                        sum_4 += temp2_r[xp+k][yp+4];
                        sum_5 += temp2_r[xp+k][yp+5];
                        sum_6 += temp2_r[xp+k][yp+6];
                        sum_7 += temp2_r[xp+k][yp+7];
                        sum_8 += temp2_r[xp+k][yp+8];
                        sum_9 += temp2_r[xp+k][yp+9];
                        sum_10 += temp2_r[xp+k][yp+10];
                        sum_11 += temp2_r[xp+k][yp+11];
                        sum_12 += temp2_r[xp+k][yp+12];
                        sum_13 += temp2_r[xp+k][yp+13];
                        sum_14 += temp2_r[xp+k][yp+14];
                        sum_15 += temp2_r[xp+k][yp+15];
                    }

                    scalar color_weight_0 = exp(-fmax(0.f, (sum_0 / neigh_r)));
                    scalar color_weight_1 = exp(-fmax(0.f, (sum_1 / neigh_r)));
                    scalar color_weight_2 = exp(-fmax(0.f, (sum_2 / neigh_r)));
                    scalar color_weight_3 = exp(-fmax(0.f, (sum_3 / neigh_r)));
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

                    // Compute final weight
                    scalar weight_0 = fmin(color_weight_0, features_weights_r[xp][yp]);
                    scalar weight_1 = fmin(color_weight_1, features_weights_r[xp][yp+1]);
                    scalar weight_2 = fmin(color_weight_2, features_weights_r[xp][yp+2]);
                    scalar weight_3 = fmin(color_weight_3, features_weights_r[xp][yp+3]);
                    scalar weight_4 = fmin(color_weight_4, features_weights_r[xp][yp+4]);
                    scalar weight_5 = fmin(color_weight_5, features_weights_r[xp][yp+5]);
                    scalar weight_6 = fmin(color_weight_6, features_weights_r[xp][yp+6]);
                    scalar weight_7 = fmin(color_weight_7, features_weights_r[xp][yp+7]);
                    scalar weight_8 = fmin(color_weight_8, features_weights_r[xp][yp+8]);
                    scalar weight_9 = fmin(color_weight_9, features_weights_r[xp][yp+9]);
                    scalar weight_10 = fmin(color_weight_10, features_weights_r[xp][yp+10]);
                    scalar weight_11 = fmin(color_weight_11, features_weights_r[xp][yp+11]);
                    scalar weight_12 = fmin(color_weight_12, features_weights_r[xp][yp+12]);
                    scalar weight_13 = fmin(color_weight_13, features_weights_r[xp][yp+13]);
                    scalar weight_14 = fmin(color_weight_14, features_weights_r[xp][yp+14]);
                    scalar weight_15 = fmin(color_weight_15, features_weights_r[xp][yp+15]);
                    
                    weight_sum[0][xp][yp] += weight_0;
                    weight_sum[0][xp][yp+1] += weight_1;
                    weight_sum[0][xp][yp+2] += weight_2;
                    weight_sum[0][xp][yp+3] += weight_3;
                    weight_sum[0][xp][yp+4] += weight_4;
                    weight_sum[0][xp][yp+5] += weight_5;
                    weight_sum[0][xp][yp+6] += weight_6;
                    weight_sum[0][xp][yp+7] += weight_7;
                    weight_sum[0][xp][yp+8] += weight_8;
                    weight_sum[0][xp][yp+9] += weight_9;
                    weight_sum[0][xp][yp+10] += weight_10;
                    weight_sum[0][xp][yp+11] += weight_11;
                    weight_sum[0][xp][yp+12] += weight_12;
                    weight_sum[0][xp][yp+13] += weight_13;
                    weight_sum[0][xp][yp+14] += weight_14;
                    weight_sum[0][xp][yp+15] += weight_15;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_r[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_r[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_r[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                        output_r[i][xp][yp+4] += weight_4 * color[i][xq][yq+4];
                        output_r[i][xp][yp+5] += weight_5 * color[i][xq][yq+5];
                        output_r[i][xp][yp+6] += weight_6 * color[i][xq][yq+6];
                        output_r[i][xp][yp+7] += weight_7 * color[i][xq][yq+7];
                        output_r[i][xp][yp+8] += weight_8 * color[i][xq][yq+8];
                        output_r[i][xp][yp+9] += weight_9 * color[i][xq][yq+9];
                        output_r[i][xp][yp+10] += weight_10 * color[i][xq][yq+10];
                        output_r[i][xp][yp+11] += weight_11 * color[i][xq][yq+11];
                        output_r[i][xp][yp+12] += weight_12 * color[i][xq][yq+12];
                        output_r[i][xp][yp+13] += weight_13 * color[i][xq][yq+13];
                        output_r[i][xp][yp+14] += weight_14 * color[i][xq][yq+14];
                        output_r[i][xp][yp+15] += weight_15 * color[i][xq][yq+15];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate G
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < img_width - R; ++xp) {
                for(int yp = R + f_g; yp < img_height - R - f_g; yp+=16) {
                    
                    scalar sum_g0 = 0.f;
                    scalar sum_g1 = 0.f;
                    scalar sum_g2 = 0.f;
                    scalar sum_g3 = 0.f;
                    scalar sum_g4 = 0.f;
                    scalar sum_g5 = 0.f;
                    scalar sum_g6 = 0.f;
                    scalar sum_g7 = 0.f;
                    scalar sum_g8 = 0.f;
                    scalar sum_g9 = 0.f;
                    scalar sum_g10 = 0.f;
                    scalar sum_g11 = 0.f;
                    scalar sum_g12 = 0.f;
                    scalar sum_g13 = 0.f;
                    scalar sum_g14 = 0.f;
                    scalar sum_g15 = 0.f;

                    for (int k=-f_g; k<=f_g; k++){
                        sum_g0 += temp[xp][yp+k];
                        sum_g1 += temp[xp][yp+k+1];
                        sum_g2 += temp[xp][yp+k+2];
                        sum_g3 += temp[xp][yp+k+3];
                        sum_g4 += temp[xp][yp+k+4];
                        sum_g5 += temp[xp][yp+k+5];
                        sum_g6 += temp[xp][yp+k+6];
                        sum_g7 += temp[xp][yp+k+7];
                        sum_g8 += temp[xp][yp+k+8];
                        sum_g9 += temp[xp][yp+k+9];
                        sum_g10 += temp[xp][yp+k+10];
                        sum_g11 += temp[xp][yp+k+11];
                        sum_g12 += temp[xp][yp+k+12];
                        sum_g13 += temp[xp][yp+k+13];
                        sum_g14 += temp[xp][yp+k+14];
                        sum_g15 += temp[xp][yp+k+15];
                    }
                    temp2_g[xp][yp] = sum_g0;
                    temp2_g[xp][yp+1] = sum_g1;
                    temp2_g[xp][yp+2] = sum_g2;
                    temp2_g[xp][yp+3] = sum_g3;
                    temp2_g[xp][yp+4] = sum_g4;
                    temp2_g[xp][yp+5] = sum_g5;
                    temp2_g[xp][yp+6] = sum_g6;
                    temp2_g[xp][yp+7] = sum_g7;
                    temp2_g[xp][yp+8] = sum_g8;
                    temp2_g[xp][yp+9] = sum_g9;
                    temp2_g[xp][yp+10] = sum_g10;
                    temp2_g[xp][yp+11] = sum_g11;
                    temp2_g[xp][yp+12] = sum_g12;
                    temp2_g[xp][yp+13] = sum_g13;
                    temp2_g[xp][yp+14] = sum_g14;
                    temp2_g[xp][yp+15] = sum_g15;
                }
            }

            // (2) Convolve along width including weighted contribution
            for(int xp = R + f_g; xp < img_width - R - f_g; ++xp) {
                for(int yp = R + f_g; yp < img_height - R - f_g; yp+=16) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final color weight
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;
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

                    for (int k=-f_g; k<=f_g; k++){
                        sum_0 += temp2_g[xp+k][yp];
                        sum_1 += temp2_g[xp+k][yp+1];
                        sum_2 += temp2_g[xp+k][yp+2];
                        sum_3 += temp2_g[xp+k][yp+3];
                        sum_4 += temp2_g[xp+k][yp+4];
                        sum_5 += temp2_g[xp+k][yp+5];
                        sum_6 += temp2_g[xp+k][yp+6];
                        sum_7 += temp2_g[xp+k][yp+7];
                        sum_8 += temp2_g[xp+k][yp+8];
                        sum_9 += temp2_g[xp+k][yp+9];
                        sum_10 += temp2_g[xp+k][yp+10];
                        sum_11 += temp2_g[xp+k][yp+11];
                        sum_12 += temp2_g[xp+k][yp+12];
                        sum_13 += temp2_g[xp+k][yp+13];
                        sum_14 += temp2_g[xp+k][yp+14];
                        sum_15 += temp2_g[xp+k][yp+15];
                    }
                    scalar color_weight_0 = exp(-fmax(0.f, (sum_0 / neigh_g)));
                    scalar color_weight_1 = exp(-fmax(0.f, (sum_1 / neigh_g)));
                    scalar color_weight_2 = exp(-fmax(0.f, (sum_2 / neigh_g)));
                    scalar color_weight_3 = exp(-fmax(0.f, (sum_3 / neigh_g)));
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
                    
                    // Compute final weight
                    scalar weight_0 = fmin(color_weight_0, features_weights_r[xp][yp]);
                    scalar weight_1 = fmin(color_weight_1, features_weights_r[xp][yp+1]);
                    scalar weight_2 = fmin(color_weight_2, features_weights_r[xp][yp+2]);
                    scalar weight_3 = fmin(color_weight_3, features_weights_r[xp][yp+3]);
                    scalar weight_4 = fmin(color_weight_4, features_weights_r[xp][yp+4]);
                    scalar weight_5 = fmin(color_weight_5, features_weights_r[xp][yp+5]);
                    scalar weight_6 = fmin(color_weight_6, features_weights_r[xp][yp+6]);
                    scalar weight_7 = fmin(color_weight_7, features_weights_r[xp][yp+7]);
                    scalar weight_8 = fmin(color_weight_8, features_weights_r[xp][yp+8]);
                    scalar weight_9 = fmin(color_weight_9, features_weights_r[xp][yp+9]);
                    scalar weight_10 = fmin(color_weight_10, features_weights_r[xp][yp+10]);
                    scalar weight_11 = fmin(color_weight_11, features_weights_r[xp][yp+11]);
                    scalar weight_12 = fmin(color_weight_12, features_weights_r[xp][yp+12]);
                    scalar weight_13 = fmin(color_weight_13, features_weights_r[xp][yp+13]);
                    scalar weight_14 = fmin(color_weight_14, features_weights_r[xp][yp+14]);
                    scalar weight_15 = fmin(color_weight_15, features_weights_r[xp][yp+15]);

                    weight_sum[1][xp][yp] += weight_0;
                    weight_sum[1][xp][yp+1] += weight_1;
                    weight_sum[1][xp][yp+2] += weight_2;
                    weight_sum[1][xp][yp+3] += weight_3;
                    weight_sum[1][xp][yp+4] += weight_4;
                    weight_sum[1][xp][yp+5] += weight_5;
                    weight_sum[1][xp][yp+6] += weight_6;
                    weight_sum[1][xp][yp+7] += weight_7;
                    weight_sum[1][xp][yp+8] += weight_8;
                    weight_sum[1][xp][yp+9] += weight_9;
                    weight_sum[1][xp][yp+10] += weight_10;
                    weight_sum[1][xp][yp+11] += weight_11;
                    weight_sum[1][xp][yp+12] += weight_12;
                    weight_sum[1][xp][yp+13] += weight_13;
                    weight_sum[1][xp][yp+14] += weight_14;
                    weight_sum[1][xp][yp+15] += weight_15;
                    
                    for (int i=0; i<3; i++){
                        output_g[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_g[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_g[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_g[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                        output_g[i][xp][yp+4] += weight_4 * color[i][xq][yq+4];
                        output_g[i][xp][yp+5] += weight_5 * color[i][xq][yq+5];
                        output_g[i][xp][yp+6] += weight_6 * color[i][xq][yq+6];
                        output_g[i][xp][yp+7] += weight_7 * color[i][xq][yq+7];
                        output_g[i][xp][yp+8] += weight_8 * color[i][xq][yq+8];
                        output_g[i][xp][yp+9] += weight_9 * color[i][xq][yq+9];
                        output_g[i][xp][yp+10] += weight_10 * color[i][xq][yq+10];
                        output_g[i][xp][yp+11] += weight_11 * color[i][xq][yq+11];
                        output_g[i][xp][yp+12] += weight_12 * color[i][xq][yq+12];
                        output_g[i][xp][yp+13] += weight_13 * color[i][xq][yq+13];
                        output_g[i][xp][yp+14] += weight_14 * color[i][xq][yq+14];
                        output_g[i][xp][yp+15] += weight_15 * color[i][xq][yq+15];
                    }
                }
            }

            // ----------------------------------------------
            // Candidate B 
            // => no color weight computation due to kc = Inf
            // ----------------------------------------------

            for(int xp = R + f_b; xp < img_width - R - f_b; ++xp) {
                for(int yp = R + f_b; yp < img_height - R - f_b; yp+=16) {

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    // Compute final weight
                    scalar weight_0 = features_weights_b[xp][yp];
                    scalar weight_1 = features_weights_b[xp][yp+1];
                    scalar weight_2 = features_weights_b[xp][yp+2];
                    scalar weight_3 = features_weights_b[xp][yp+3];
                    scalar weight_4 = features_weights_b[xp][yp+4];
                    scalar weight_5 = features_weights_b[xp][yp+5];
                    scalar weight_6 = features_weights_b[xp][yp+6];
                    scalar weight_7 = features_weights_b[xp][yp+7];
                    scalar weight_8 = features_weights_b[xp][yp+8];
                    scalar weight_9 = features_weights_b[xp][yp+9];
                    scalar weight_10 = features_weights_b[xp][yp+10];
                    scalar weight_11 = features_weights_b[xp][yp+11];
                    scalar weight_12 = features_weights_b[xp][yp+12];
                    scalar weight_13 = features_weights_b[xp][yp+13];
                    scalar weight_14 = features_weights_b[xp][yp+14];
                    scalar weight_15 = features_weights_b[xp][yp+15];
                    
                    weight_sum[2][xp][yp] += weight_0;
                    weight_sum[2][xp][yp+1] += weight_1;
                    weight_sum[2][xp][yp+2] += weight_2;
                    weight_sum[2][xp][yp+3] += weight_3;
                    weight_sum[2][xp][yp+4] += weight_4;
                    weight_sum[2][xp][yp+5] += weight_5;
                    weight_sum[2][xp][yp+6] += weight_6;
                    weight_sum[2][xp][yp+7] += weight_7;
                    weight_sum[2][xp][yp+8] += weight_8;
                    weight_sum[2][xp][yp+9] += weight_9;
                    weight_sum[2][xp][yp+10] += weight_10;
                    weight_sum[2][xp][yp+11] += weight_11;
                    weight_sum[2][xp][yp+12] += weight_12;
                    weight_sum[2][xp][yp+13] += weight_13;
                    weight_sum[2][xp][yp+14] += weight_14;
                    weight_sum[2][xp][yp+15] += weight_15;
                    
                    for (int i=0; i<3; i++){
                        output_b[i][xp][yp] += weight_0 * color[i][xq][yq];
                        output_b[i][xp][yp+1] += weight_1 * color[i][xq][yq+1];
                        output_b[i][xp][yp+2] += weight_2 * color[i][xq][yq+2];
                        output_b[i][xp][yp+3] += weight_3 * color[i][xq][yq+3];
                        output_b[i][xp][yp+4] += weight_4 * color[i][xq][yq+4];
                        output_b[i][xp][yp+5] += weight_5 * color[i][xq][yq+5];
                        output_b[i][xp][yp+6] += weight_6 * color[i][xq][yq+6];
                        output_b[i][xp][yp+7] += weight_7 * color[i][xq][yq+7];
                        output_b[i][xp][yp+8] += weight_8 * color[i][xq][yq+8];
                        output_b[i][xp][yp+9] += weight_9 * color[i][xq][yq+9];
                        output_b[i][xp][yp+10] += weight_10 * color[i][xq][yq+10];
                        output_b[i][xp][yp+11] += weight_11 * color[i][xq][yq+11];
                        output_b[i][xp][yp+12] += weight_12 * color[i][xq][yq+12];
                        output_b[i][xp][yp+13] += weight_13 * color[i][xq][yq+13];
                        output_b[i][xp][yp+14] += weight_14 * color[i][xq][yq+14];
                        output_b[i][xp][yp+15] += weight_15 * color[i][xq][yq+15];
                    }
                }
            }

        }
    }

    // Final Weight Normalization R
    for(int xp = R + f_r; xp < img_width - R - f_r; ++xp) {
        for(int yp = R + f_r; yp < img_height - R - f_r; yp+=16) {
        
            scalar w_0 = weight_sum[0][xp][yp];
            scalar w_1 = weight_sum[0][xp][yp+1];
            scalar w_2 = weight_sum[0][xp][yp+2];
            scalar w_3 = weight_sum[0][xp][yp+3];
            scalar w_4 = weight_sum[0][xp][yp+4];
            scalar w_5 = weight_sum[0][xp][yp+5];
            scalar w_6 = weight_sum[0][xp][yp+6];
            scalar w_7 = weight_sum[0][xp][yp+7];
            scalar w_8 = weight_sum[0][xp][yp+8];
            scalar w_9 = weight_sum[0][xp][yp+9];
            scalar w_10 = weight_sum[0][xp][yp+10];
            scalar w_11 = weight_sum[0][xp][yp+11];
            scalar w_12 = weight_sum[0][xp][yp+12];
            scalar w_13 = weight_sum[0][xp][yp+13];
            scalar w_14 = weight_sum[0][xp][yp+14];
            scalar w_15 = weight_sum[0][xp][yp+15];

            for (int i=0; i<3; i++){
                output_r[i][xp][yp] /= w_0;
                output_r[i][xp][yp+1] /= w_1;
                output_r[i][xp][yp+2] /= w_2;
                output_r[i][xp][yp+3] /= w_3;
                output_r[i][xp][yp+4] /= w_4;
                output_r[i][xp][yp+5] /= w_5;
                output_r[i][xp][yp+6] /= w_6;
                output_r[i][xp][yp+7] /= w_7;
                output_r[i][xp][yp+8] /= w_8;
                output_r[i][xp][yp+9] /= w_9;
                output_r[i][xp][yp+10] /= w_10;
                output_r[i][xp][yp+11] /= w_11;
                output_r[i][xp][yp+12] /= w_12;
                output_r[i][xp][yp+13] /= w_13;
                output_r[i][xp][yp+14] /= w_14;
                output_r[i][xp][yp+15] /= w_15;
            }
        }
    }

    // Final Weight Normalization G
   for(int xp = R + f_g; xp < img_width - R - f_g; ++xp) {
        for(int yp = R + f_g; yp < img_height - R - f_g; yp+=16) {
        
            scalar w_0 = weight_sum[1][xp][yp];
            scalar w_1 = weight_sum[1][xp][yp+1];
            scalar w_2 = weight_sum[1][xp][yp+2];
            scalar w_3 = weight_sum[1][xp][yp+3];
            scalar w_4 = weight_sum[1][xp][yp+4];
            scalar w_5 = weight_sum[1][xp][yp+5];
            scalar w_6 = weight_sum[1][xp][yp+6];
            scalar w_7 = weight_sum[1][xp][yp+7];
            scalar w_8 = weight_sum[1][xp][yp+8];
            scalar w_9 = weight_sum[1][xp][yp+9];
            scalar w_10 = weight_sum[1][xp][yp+10];
            scalar w_11 = weight_sum[1][xp][yp+11];
            scalar w_12 = weight_sum[1][xp][yp+12];
            scalar w_13 = weight_sum[1][xp][yp+13];
            scalar w_14 = weight_sum[1][xp][yp+14];
            scalar w_15 = weight_sum[1][xp][yp+15];

            for (int i=0; i<3; i++){
                output_g[i][xp][yp] /= w_0;
                output_g[i][xp][yp+1] /= w_1;
                output_g[i][xp][yp+2] /= w_2;
                output_g[i][xp][yp+3] /= w_3;
                output_g[i][xp][yp+4] /= w_4;
                output_g[i][xp][yp+5] /= w_5;
                output_g[i][xp][yp+6] /= w_6;
                output_g[i][xp][yp+7] /= w_7;
                output_g[i][xp][yp+8] /= w_8;
                output_g[i][xp][yp+9] /= w_9;
                output_g[i][xp][yp+10] /= w_10;
                output_g[i][xp][yp+11] /= w_11;
                output_g[i][xp][yp+12] /= w_12;
                output_g[i][xp][yp+13] /= w_13;
                output_g[i][xp][yp+14] /= w_14;
                output_g[i][xp][yp+15] /= w_15;
            }
        }
    }

    // Final Weight Normalization B
   for(int xp = R + f_b; xp < img_width - R - f_b; ++xp) {
        for(int yp = R + f_b; yp < img_height - R - f_b; yp+=16) {
        
            scalar w_0 = weight_sum[2][xp][yp];
            scalar w_1 = weight_sum[2][xp][yp+1];
            scalar w_2 = weight_sum[2][xp][yp+2];
            scalar w_3 = weight_sum[2][xp][yp+3];
            scalar w_4 = weight_sum[2][xp][yp+4];
            scalar w_5 = weight_sum[2][xp][yp+5];
            scalar w_6 = weight_sum[2][xp][yp+6];
            scalar w_7 = weight_sum[2][xp][yp+7];
            scalar w_8 = weight_sum[2][xp][yp+8];
            scalar w_9 = weight_sum[2][xp][yp+9];
            scalar w_10 = weight_sum[2][xp][yp+10];
            scalar w_11 = weight_sum[2][xp][yp+11];
            scalar w_12 = weight_sum[2][xp][yp+12];
            scalar w_13 = weight_sum[2][xp][yp+13];
            scalar w_14 = weight_sum[2][xp][yp+14];
            scalar w_15 = weight_sum[2][xp][yp+15];

            for (int i=0; i<3; i++){
                output_b[i][xp][yp] /= w_0;
                output_b[i][xp][yp+1] /= w_1;
                output_b[i][xp][yp+2] /= w_2;
                output_b[i][xp][yp+3] /= w_3;
                output_b[i][xp][yp+4] /= w_4;
                output_b[i][xp][yp+5] /= w_5;
                output_b[i][xp][yp+6] /= w_6;
                output_b[i][xp][yp+7] /= w_7;
                output_b[i][xp][yp+8] /= w_8;
                output_b[i][xp][yp+9] /= w_9;
                output_b[i][xp][yp+10] /= w_10;
                output_b[i][xp][yp+11] /= w_11;
                output_b[i][xp][yp+12] /= w_12;
                output_b[i][xp][yp+13] /= w_13;
                output_b[i][xp][yp+14] /= w_14;
                output_b[i][xp][yp+15] /= w_15;
            }
        }
    }

    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < R + f_r; yp++){
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_r; xp++){
            for (int yp = R + f_r ; yp < img_height - R - f_r; yp++){
            
                output_r[i][xp][yp] = color[i][xp][yp];
                output_r[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
                output_b[i][xp][yp] = color[i][xp][yp];
                output_b[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
             }
        }
    }

    // Candidate SECOND since f_g != f_r
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < R + f_g; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][xp][img_height - yp - 1] = color[i][xp][img_height - yp - 1];
            }
        }
        for(int xp = 0; xp < R + f_g; xp++){
            for (int yp = R + f_g ; yp < img_height - R - f_g; yp++){
                output_g[i][xp][yp] = color[i][xp][yp];
                output_g[i][img_width - xp - 1][yp] = color[i][img_width - xp - 1][yp];
            }
        }
    }

    // Free memory
    free_buffer(&weight_sum, img_width);
    free_channel(&temp, img_width);
    free_channel(&temp2_r, img_width);
    free_channel(&temp2_g, img_width);
    free_channel(&features_weights_r, img_width);
    free_channel(&features_weights_b, img_width);
    free_buffer(&gradients, img_width);

}
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flt.hpp"
#include "flt_restructure.hpp"
#include "memory_mgmt.hpp"

// Get parameters
#define F_R 1
#define F_G 3
#define F_B 1
#define TAU_R 0.001f
#define TAU_G 0.001f
#define TAU_B 0.0001f
#define KC_SQUARED_R 4.0f
#define KF_SQUARED 0.36f
#define KC_SQUARED_G 4.0f

#define TAU_KF_R 1.0f/(TAU_R*KF_SQUARED)
#define TAU_KF_B 1.0f/(TAU_B*KF_SQUARED)

// $auto_unroll
inline __attribute__((always_inline)) void compute_denominators(scalar *denominators, buffer features, buffer features_var, const int R, const int W, const int H) {
    const int WH = W*H;
    
    scalar featC0, featL0, featR0, featU0, featD0, feat_var0;
    scalar diffL0, diffR0, diffU0, diffD0;
    scalar sqdiffL0, sqdiffR0, sqdiffU0, sqdiffD0;
    scalar gradH0, gradV0, grad0;
    // scalar maxi_feat, inverted_maxi_feat;
    
    for(int i=0; i<NB_FEATURES;++i) {
        for(int x =  R+F_R; x < W - R - F_R; ++x) {
            for(int y =  R+F_R; y < H -  R - F_R; y += 1) {

                // Read
                featC0 = features[i][x][y + 0];
                featL0 = features[i][x-1][y + 0];
                featR0 = features[i][x+1][y + 0];
                featU0 = features[i][x][y+1 + 0];
                featD0 = features[i][x][y-1 + 0];
                //feat_var = features_var[i][x][y];
                
                // Compute gradients
                diffL0 = featC0 - featL0;
                diffR0 = featC0 - featR0;
                diffU0 = featC0 - featU0;
                diffD0 = featC0 - featD0;

                sqdiffL0 = diffL0 * diffL0;
                sqdiffR0 = diffR0 * diffR0;
                sqdiffU0 = diffU0 * diffU0;
                sqdiffD0 = diffD0 * diffD0;
                
                gradH0 = fmin(sqdiffL0, sqdiffR0);
                gradV0 = fmin(sqdiffU0, sqdiffD0);

                grad0 = gradV0 + gradH0;

                // Compute denominator
                //maxi_feat = KF_SQUARED * fmax(feat_var, grad);
                //inverted_maxi_feat = 1.0f / maxi_feat;

                // Store
                denominators[WH*i + x * W + y + 0] = grad0;
            
            } 
        }
    }
}
// $end_unroll

inline __attribute__((always_inline)) void color_weights_ILP2(scalar *temp, buffer color, buffer color_var, const int r_x, const int r_y, const int W, const int H, const int R) {
    scalar colpr, colqr, varpr, varqr;
    scalar colpg, colqg, varpg, varqg;
    scalar colpb, colqb, varpb, varqb;
    scalar distr, sq_distr;
    scalar distg, sq_distg;
    scalar distb, sq_distb;
    scalar min_varr, var_cancelr, var_termr, dist_varr;
    scalar min_varg, var_cancelg, var_termg, dist_varg;
    scalar min_varb, var_cancelb, var_termb, dist_varb;
    scalar normalizationr, color_weightr;
    scalar normalizationg, color_weightg;
    scalar normalizationb, color_weightb;
    scalar sum_weights;
    
    // Still easy to vectorize, just consider each color[i] as a different buffer
    for(int xp = R; xp < W - R; ++xp) {
        for(int yp = R; yp < H - R; ++yp) {
            int xq = xp + r_x;
            int yq = yp + r_y;

            // Read
            colpr = color[0][xp][yp];
            colqr = color[0][xq][yq];
            varpr = color_var[0][xp][yp];
            varqr = color_var[0][xq][yq];
            colpg = color[1][xp][yp];
            colqg = color[1][xq][yq];
            varpg = color_var[1][xp][yp];
            varqg = color_var[1][xq][yq];
            colpb = color[2][xp][yp];
            colqb = color[2][xq][yq];
            varpb = color_var[2][xp][yp];
            varqb = color_var[2][xq][yq];

            // Compute
            distr = colpr - colqr;
            distg = colpg - colqg;
            distb = colpb - colqb;
            sq_distr = distr * distr;
            sq_distg = distg * distg;
            sq_distb = distb * distb;

            min_varr = fmin(varpr, varqr);
            min_varg = fmin(varpg, varqg);
            min_varb = fmin(varpb, varqb);
            var_cancelr = varpr + min_varr;
            var_cancelg = varpg + min_varg;
            var_cancelb = varpb + min_varb;
            
            var_termr = varpr + varqr;
            var_termg = varpg + varqg;
            var_termb = varpb + varqb;
            normalizationr = EPSILON + KC_SQUARED_R*var_termr;
            normalizationg = EPSILON + KC_SQUARED_R*var_termg;
            normalizationb = EPSILON + KC_SQUARED_R*var_termb;

            dist_varr = var_cancelr - sq_distr;
            dist_varg = var_cancelg - sq_distg;
            dist_varb = var_cancelb - sq_distb;
            color_weightr = dist_varr / normalizationr;
            color_weightg = dist_varg / normalizationg;
            color_weightb = dist_varb / normalizationb;

            sum_weights = color_weightr + color_weightg + color_weightb;
            
            // Store
            temp[xp * W + yp] = sum_weights;

        }
    }
}

// $unroll 3
inline __attribute__((always_inline)) void precompute_features_ILP2(scalar *features_weights_r, scalar *features_weights_b, scalar *weight_sum_b, buffer color, buffer features, buffer features_var, scalar *denominators, const int r_x, const int r_y, const int R, const int W, const int H) {
    const int WH = W*H;

    scalar df_r0 = 0.f, df_b0 = 0.f;
    scalar df_r1 = 0.f, df_b1 = 0.f;
    scalar df_r2 = 0.f, df_b2 = 0.f;
    scalar fp0, fq0, varp0, varq0;
    scalar fp1, fq1, varp1, varq1;
    scalar fp2, fq2, varp2, varq2;
    scalar grad0;
    scalar grad1;
    scalar grad2;
    scalar dist0, sqdist0, varpq0, var_cancel0, var_max0, normalization_b0, normalization_r0, dist_var0;
    scalar dist1, sqdist1, varpq1, var_cancel1, var_max1, normalization_b1, normalization_r1, dist_var1;
    scalar dist2, sqdist2, varpq2, var_cancel2, var_max2, normalization_b2, normalization_r2, dist_var2;

    
    for(int xp = R + F_R; xp < W - R - F_R; ++xp) {
        for(int yp = R + F_R; yp < H - R - F_R; ++yp) {
            df_r0 = 0.f; df_b0 = 0.f;
            df_r1 = 0.f; df_b1 = 0.f;
            df_r2 = 0.f; df_b2 = 0.f;
            int xq = xp + r_x;
            int yq = yp + r_y;
            for(int j=0; j<NB_FEATURES;j += 3){
                
                fp0 = features[j][xp][yp + 0];
                fp1 = features[j][xp][yp + 1];
                fp2 = features[j][xp][yp + 2];
                fq0 = features[j][xq][yq + 0];
                fq1 = features[j][xq][yq + 1];
                fq2 = features[j][xq][yq + 2];
                varp0 = features_var[j][xp][yp + 0];
                varp1 = features_var[j][xp][yp + 1];
                varp2 = features_var[j][xp][yp + 2];
                varq0 = features_var[j][xq][yq + 0];
                varq1 = features_var[j][xq][yq + 1];
                varq2 = features_var[j][xq][yq + 2];
                grad0 = denominators[j * WH + xp * W + yp + 0];
                grad1 = denominators[j * WH + xp * W + yp + 1];
                grad2 = denominators[j * WH + xp * W + yp + 2];
                
                dist0 = fp0 - fq0;
                dist1 = fp1 - fq1;
                dist2 = fp2 - fq2;
                varpq0 = fmin(varp0, varq0);
                varpq1 = fmin(varp1, varq1);
                varpq2 = fmin(varp2, varq2);
                var_cancel0 = varp0 + varpq0;
                var_cancel1 = varp1 + varpq1;
                var_cancel2 = varp2 + varpq2;
                sqdist0 = dist0 * dist0;
                sqdist1 = dist1 * dist1;
                sqdist2 = dist2 * dist2;
                dist_var0 = var_cancel0 - sqdist0;
                dist_var1 = var_cancel1 - sqdist1;
                dist_var2 = var_cancel2 - sqdist2;

                // ============ !!!!!!! =================================================================
                // ToDo: Precompute normalization constants => always the same independet of R
                // @Comment from Nino: Not successfull so far => same runtime but less flops => yet less performance
                scalar var_max0 = fmax(varp0, grad0);
                scalar var_max1 = fmax(varp1, grad1);
                scalar var_max2 = fmax(varp2, grad2);
                scalar normalization_r0 = KF_SQUARED*fmax(TAU_R, var_max0);
                scalar normalization_r1 = KF_SQUARED*fmax(TAU_R, var_max1);
                scalar normalization_r2 = KF_SQUARED*fmax(TAU_R, var_max2);
                scalar normalization_b0 = KF_SQUARED*fmax(TAU_B, var_max0);
                scalar normalization_b1 = KF_SQUARED*fmax(TAU_B, var_max1);
                scalar normalization_b2 = KF_SQUARED*fmax(TAU_B, var_max2);
                // ============ !!!!!!! =================================================================

                //df_r = fmin(df_r, dist_var/norm_r[j * WH + xp * W + yp]);
                //df_b = fmin(df_b, dist_var/norm_b[j * WH + xp * W + yp]);
                df_r0 = fmin(df_r0, (dist_var0)/normalization_r0);
                df_r1 = fmin(df_r1, (dist_var1)/normalization_r1);
                df_r2 = fmin(df_r2, (dist_var2)/normalization_r2);
                df_b0 = fmin(df_b0, (dist_var0)/normalization_b0);
                df_b1 = fmin(df_b1, (dist_var1)/normalization_b1);
                df_b2 = fmin(df_b2, (dist_var2)/normalization_b2);
            } 
            features_weights_r[xp * W + yp + 0] = df_r0;
            features_weights_r[xp * W + yp + 1] = df_r1;
            features_weights_r[xp * W + yp + 2] = df_r2;
            features_weights_b[xp * W + yp + 0] = df_b0;
            features_weights_b[xp * W + yp + 1] = df_b1;
            features_weights_b[xp * W + yp + 2] = df_b2;
        }
    }
}
// $end_unroll

void candidate_filtering_all_ILP2(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, const int R, const int W, const int H){

    const int WH = W*H;

    // Handling Inner Part   
    // -------------------

    // Allocate buffer weights_sum for normalizing
    scalar* weight_sum_r;
    scalar* weight_sum_g;
    scalar* weight_sum_b;

    weight_sum_r = (scalar*) calloc(WH, sizeof(scalar));
    weight_sum_g = (scalar*) calloc(WH, sizeof(scalar));
    weight_sum_b = (scalar*) calloc(WH, sizeof(scalar));

    // Init temp channel
    scalar* temp;
    scalar* temp2_r;
    scalar* temp2_g;
    temp = (scalar*) malloc(WH * sizeof(scalar));
    temp2_r = (scalar*) malloc(WH * sizeof(scalar));
    temp2_g = (scalar*) malloc(WH * sizeof(scalar));

    // Allocate feature weights buffer
    scalar* features_weights_r;
    scalar* features_weights_b;
    features_weights_r = (scalar*) malloc(WH * sizeof(scalar));
    features_weights_b = (scalar*) malloc(WH * sizeof(scalar));
    
    // Compute gradients and precompute divs
    scalar *denominators;
    denominators = (scalar*) malloc(3 * WH * sizeof(scalar));

    compute_denominators(denominators, features, features_var, R, W, H);


    // Precompute size of neighbourhood
    scalar neigh_r_inv = 1. / (3*(2*F_R+1)*(2*F_R+1));
    scalar neigh_g_inv = 1. / (3*(2*F_G+1)*(2*F_G+1));

    // Covering the neighbourhood
    for (int r_x = -R; r_x <= R; r_x++){
        for (int r_y = -R; r_y <= R; r_y++){
        
            // #######################################################################################
            // WEIGHT COMPUTATION
            // #######################################################################################

            // Compute Color Weight for all pixels with fixed r
            color_weights_ILP2(temp, color, color_var, r_x, r_y, W, H, R);

           
            // Precompute feature weights and compute candidate B
            precompute_features_ILP2(features_weights_r, features_weights_b, weight_sum_b, color, features, features_var, denominators, r_x, r_y, R, W, H);

            

            // #######################################################################################
            // BOX FILTERING => seperability of box filter kernel => two linear operations
            // #######################################################################################
            
            // ----------------------------------------------
            // Candidate R
            // ----------------------------------------------
            // (1) Convolve along height
            for(int xp = R; xp < W - R; ++xp) {
                for(int yp = R + F_R; yp < H - R - F_R; yp+=8) {
                    scalar sum_r_0 = 0.f;
                    scalar sum_r_1 = 0.f;
                    scalar sum_r_2 = 0.f;
                    scalar sum_r_3 = 0.f;
                    scalar sum_r_4 = 0.f;
                    scalar sum_r_5 = 0.f;
                    scalar sum_r_6 = 0.f;
                    scalar sum_r_7 = 0.f;

                    for (int k=-F_R; k<=F_R; k++){
                        sum_r_0 += temp[xp * W + yp+k];
                        sum_r_1 += temp[xp * W + yp+k+1];
                        sum_r_2 += temp[xp * W + yp+k+2];
                        sum_r_3 += temp[xp * W + yp+k+3];
                        sum_r_4 += temp[xp * W + yp+k+4];
                        sum_r_5 += temp[xp * W + yp+k+5];
                        sum_r_6 += temp[xp * W + yp+k+6];
                        sum_r_7 += temp[xp * W + yp+k+7];
                    }
                    temp2_r[xp * W + yp] = sum_r_0;
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
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    // Unrolled Summation => Fixed for f_r=1 => 2*f+1 = 3
                    sum_0 += temp2_r[(xp-1) * W + yp];    
                    sum_0 += temp2_r[(xp) * W + yp];
                    sum_0 += temp2_r[(xp+1) * W + yp];

                    sum_1 += temp2_r[(xp-1) * W + yp+1];
                    sum_1 += temp2_r[(xp) * W + yp+1];
                    sum_1 += temp2_r[(xp+1) * W + yp+1];

                    sum_2 += temp2_r[(xp-1) * W + yp+2];
                    sum_2 += temp2_r[(xp) * W + yp+2];
                    sum_2 += temp2_r[(xp+1) * W + yp+2];
   
                    sum_3 += temp2_r[(xp-1) * W + yp+3];
                    sum_3 += temp2_r[(xp) * W + yp+3];
                    sum_3 += temp2_r[(xp+1) * W + yp+3];


                    // Compute Color Weight
                    scalar color_weight_0 = (sum_0 * neigh_r_inv);
                    scalar color_weight_1 = (sum_1 * neigh_r_inv);
                    scalar color_weight_2 = (sum_2 * neigh_r_inv);
                    scalar color_weight_3 = (sum_3 * neigh_r_inv);

                    // Compute final weight
                    scalar weight_0 = exp(fmin(color_weight_0, features_weights_r[xp * W + yp]));
                    scalar weight_1 = exp(fmin(color_weight_1, features_weights_r[xp * W + yp+1]));
                    scalar weight_2 = exp(fmin(color_weight_2, features_weights_r[xp * W + yp+2]));
                    scalar weight_3 = exp(fmin(color_weight_3, features_weights_r[xp * W + yp+3]));

                    weight_sum_r[xp * W + yp] += weight_0;
                    weight_sum_r[xp * W + yp+1] += weight_1;
                    weight_sum_r[xp * W + yp+2] += weight_2;
                    weight_sum_r[xp * W + yp+3] += weight_3;
                    
                    for (int i=0; i<3; i++){
                        output_r[i][xp][yp] += weight_0 * color[i][xq][yq];
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
                    
                    scalar sum_g_0 = 0.f;
                    scalar sum_g_1 = 0.f;
                    scalar sum_g_2 = 0.f;
                    scalar sum_g_3 = 0.f;
                    scalar sum_g_4 = 0.f;
                    scalar sum_g_5 = 0.f;
                    scalar sum_g_6 = 0.f;
                    scalar sum_g_7 = 0.f;

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
                    scalar sum_0 = 0.f;
                    scalar sum_1 = 0.f;
                    scalar sum_2 = 0.f;
                    scalar sum_3 = 0.f;

                    // Unrolled Summation => Fixed for f_g=3 => 2*f_g+1 = 7
                    sum_0 += temp2_g[(xp-3) * W + yp];
                    sum_0 += temp2_g[(xp-2) * W + yp];
                    sum_0 += temp2_g[(xp-1) * W + yp];
                    sum_0 += temp2_g[(xp) * W + yp];
                    sum_0 += temp2_g[(xp+1) * W + yp];
                    sum_0 += temp2_g[(xp+2) * W + yp];
                    sum_0 += temp2_g[(xp+3) * W + yp];

                    sum_1 += temp2_g[(xp-3) * W + yp+1];
                    sum_1 += temp2_g[(xp-2) * W + yp+1];
                    sum_1 += temp2_g[(xp-1) * W + yp+1];
                    sum_1 += temp2_g[(xp) * W + yp+1];
                    sum_1 += temp2_g[(xp+1) * W + yp+1];
                    sum_1 += temp2_g[(xp+2) * W + yp+1];
                    sum_1 += temp2_g[(xp+3) * W + yp+1];

                    sum_2 += temp2_g[(xp-3) * W + yp+2];
                    sum_2 += temp2_g[(xp-2) * W + yp+2];
                    sum_2 += temp2_g[(xp-1) * W + yp+2];
                    sum_2 += temp2_g[(xp) * W + yp+2];
                    sum_2 += temp2_g[(xp+1) * W + yp+2];
                    sum_2 += temp2_g[(xp+2) * W + yp+2];
                    sum_2 += temp2_g[(xp+3) * W + yp+2];

                    sum_3 += temp2_g[(xp-3) * W + yp+3];
                    sum_3 += temp2_g[(xp-2) * W + yp+3];
                    sum_3 += temp2_g[(xp-1) * W + yp+3];
                    sum_3 += temp2_g[(xp) * W + yp+3];
                    sum_3 += temp2_g[(xp+1) * W + yp+3];
                    sum_3 += temp2_g[(xp+2) * W + yp+3];
                    sum_3 += temp2_g[(xp+3) * W + yp+3];

                    // Compute Color Weight
                    scalar color_weight_0 = (sum_0 * neigh_g_inv);
                    scalar color_weight_1 = (sum_1 * neigh_g_inv);
                    scalar color_weight_2 = (sum_2 * neigh_g_inv);
                    scalar color_weight_3 = (sum_3 * neigh_g_inv);
                    
                    // Compute final weight
                    scalar weight_0 = exp(fmin(color_weight_0, features_weights_r[xp * W + yp]));
                    scalar weight_1 = exp(fmin(color_weight_1, features_weights_r[xp * W + yp+1]));
                    scalar weight_2 = exp(fmin(color_weight_2, features_weights_r[xp * W + yp+2]));
                    scalar weight_3 = exp(fmin(color_weight_3, features_weights_r[xp * W + yp+3]));

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

                    int xq = xp + r_x;
                    int yq = yp + r_y;

                    scalar weight_0 = exp(features_weights_b[xp * W + yp]);
                    scalar weight_1 = exp(features_weights_b[xp * W + yp+1]);
                    scalar weight_2 = exp(features_weights_b[xp * W + yp+2]);
                    scalar weight_3 = exp(features_weights_b[xp * W + yp+3]);

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
        for(int yp = R + F_G; yp < H - R - F_G; yp+=4) {
        
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
        for(int yp = R + F_B; yp < H - R - F_B; yp+=4) {
        
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


    // Handline Border Cases 
    // ----------------------------------
    // Candidate FIRST and THIRD (due to f_r = f_b)
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

    // Candidate SECOND since f_g != f_r
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
    free(denominators);

}

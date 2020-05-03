#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flt_opt1.hpp"
#include "../../memory_mgmt.hpp"




void sure_Basic(channel output, buffer c, buffer c_var, buffer cand, buffer cand_d, int img_width, int img_height){

    scalar d, v;
    
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){

            scalar sure = 0.f;

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Calculate terms
                d = cand[i][x][y] - c[i][x][y];
                d *= d;
                v = c_var[i][x][y];
                v *= v;

                // Summing up
                sure += d - v + (2 * v * cand_d[i][x][y]); 

            }
            // Store sure error estimate
            output[x][y] = sure;
        }
    }
}

void sure_opt1(channel sure_r, channel sure_g, channel sure_b, buffer c, buffer c_var, buffer r, buffer d_r, buffer g, buffer d_g, buffer b, buffer d_b, int img_width, int img_height){

    scalar dr, dg, db, v;
    
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){

            scalar sure__r = 0.f;
            scalar sure__g = 0.f;
            scalar sure__b = 0.f;

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Calculate terms
                const scalar c_ixy = c[i][x][y];
                const scalar c_var_ixy_sqr = c_var[i][x][y]*c_var[i][x][y];
                const scalar c_var_ixy_sqr_dbl = 2*c_var_ixy_sqr;

                dr = r[i][x][y] - c_ixy;
                dr *= dr;
                dg = g[i][x][y] - c_ixy;
                dg *= dg;
                db = b[i][x][y] - c_ixy;
                db *= db;

                // Summing up
                sure__r += dr - c_var_ixy_sqr + (c_var_ixy_sqr_dbl * d_r[i][x][y]);
                sure__g += dg - c_var_ixy_sqr + (c_var_ixy_sqr_dbl * d_g[i][x][y]); 
                sure__b += db - c_var_ixy_sqr + (c_var_ixy_sqr_dbl * d_b[i][x][y]); 

            }
            // Store sure error estimate
            sure_r[x][y] = sure__r;
            sure_g[x][y] = sure__g;
            sure_b[x][y] = sure__b;
        }
    }
}



void flt_buffer_Basic(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
            }
        }
        for (int yp = p.r+p.f ; yp < img_height - p.r+p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
            }
        }

    }

    
    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width - p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height - p.r-p.f; ++yp) {

            sum_weights = 0.f;

            // Init output to 0 => TODO: maybe we can do this with calloc
            for(int i = 0; i < 3; ++i)
                output[i][xp][yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight_opt1(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;

                    // Add contribution term
                    for(int i=0;i<3;++i)
                        output[i][xp][yp] += input[i][xq][yq] * wc;
                }
            }

            // Normalization step
            for(int i=0;i<3;++i)
                output[i][xp][yp] /= (sum_weights + EPSILON);
        }
    }
}

void flt_buffer_opt1(buffer f_filtered, buffer f_var_filtered, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                f_filtered[i][xp][yp] = f[i][xp][yp];
                f_filtered[i][xp][img_height - yp - 1] = f[i][xp][img_height - yp - 1];

                f_var_filtered[i][xp][yp] = var_f[i][xp][yp];
                f_var_filtered[i][xp][img_height - yp - 1] = var_f[i][xp][img_height - yp - 1];
            }
        }
        for (int yp = p.r+p.f ; yp < img_height - p.r+p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                f_filtered[i][xp][yp] = f[i][xp][yp];
                f_filtered[i][img_width - xp - 1][yp] = f[i][img_width - xp - 1][yp];

                f_var_filtered[i][xp][yp] = var_f[i][xp][yp];
                f_var_filtered[i][img_width - xp - 1][yp] = var_f[i][img_width - xp - 1][yp];
            }
        }

    }

    
    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width - p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height - p.r-p.f; ++yp) {

            sum_weights = EPSILON;

            // Init output to 0 => TODO: maybe we can do this with calloc
            for(int i = 0; i < 3; ++i){
                f_filtered[i][xp][yp] = 0.f;
                f_var_filtered[i][xp][yp] = 0.f;
            }

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight_opt1(f, var_f, p, xp, yp, xq, yq);
                    sum_weights += wc;

                    // Add contribution term
                    for(int i=0;i<3;++i){
                        f_filtered[i][xp][yp] += f[i][xq][yq] * wc;
                        f_var_filtered[i][xp][yp] += var_f[i][xq][yq] * wc;
                    }
                }
            }

            // Normalization step
            for(int i=0;i<3;++i){
                f_filtered[i][xp][yp] /= sum_weights;
                f_var_filtered[i][xp][yp] /= sum_weights;
            }
        }
    }
}

void flt_channel_Basic(channel output, channel input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < img_width; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            output[xp][yp] = input[xp][yp];
            output[xp][img_height - yp - 1] = input[xp][img_height - yp - 1];
        }
    }
    for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
        for(int xp = 0; xp < p.r+p.f; xp++){
            output[xp][yp] = input[xp][yp];
            output[img_width - xp - 1][yp] = input[img_width - xp - 1][yp];
        }
    }


    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            sum_weights = 0.f;

            // Init output to 0 => TODO: maybe we can do this with calloc
            output[xp][yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight_opt1(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;

                    // Add contribution term
                    output[xp][yp] += input[xq][yq] * wc;
                }
            }

            // Normalization step
            output[xp][yp] /= (sum_weights + EPSILON);
        }
    }


}


void flt_channel_opt1(channel output_1, channel input_1, channel output_2, channel input_2, channel output_3, channel input_3, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < img_width; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            output_1[xp][yp] = input_1[xp][yp];
            output_1[xp][img_height - yp - 1] = input_1[xp][img_height - yp - 1];
            output_2[xp][yp] = input_2[xp][yp];
            output_2[xp][img_height - yp - 1] = input_2[xp][img_height - yp - 1];
            output_3[xp][yp] = input_3[xp][yp];
            output_3[xp][img_height - yp - 1] = input_3[xp][img_height - yp - 1];
        }
    }
    for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
        for(int xp = 0; xp < p.r+p.f; xp++){
            output_1[xp][yp] = input_1[xp][yp];
            output_1[img_width - xp - 1][yp] = input_1[img_width - xp - 1][yp];
            output_2[xp][yp] = input_2[xp][yp];
            output_2[img_width - xp - 1][yp] = input_2[img_width - xp - 1][yp];
            output_3[xp][yp] = input_3[xp][yp];
            output_3[img_width - xp - 1][yp] = input_3[img_width - xp - 1][yp];
        }
    }


    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            sum_weights = EPSILON;

            // Init output_1 to 0 => TODO: maybe we can do this with calloc
            output_1[xp][yp] = 0.f;
            output_2[xp][yp] = 0.f;
            output_3[xp][yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight_opt1(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;

                    // Add contribution term
                    output_1[xp][yp] += input_1[xq][yq] * wc;
                    output_2[xp][yp] += input_2[xq][yq] * wc;
                    output_3[xp][yp] += input_3[xq][yq] * wc;
                }
            }

            // Normalization step
            output_1[xp][yp] /= sum_weights;
            output_2[xp][yp] /= sum_weights;
            output_3[xp][yp] /= sum_weights;
        }
    }
}

void flt_channel_opt1_sel(channel output_1, channel input_1, channel output_2, channel input_2, channel output_3, channel input_3, channel sel_r, channel sel_g, channel sel_b, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < img_width; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            output_1[xp][yp] = input_1[xp][yp];
            output_1[xp][img_height - yp - 1] = input_1[xp][img_height - yp - 1];
            output_2[xp][yp] = input_2[xp][yp];
            output_2[xp][img_height - yp - 1] = input_2[xp][img_height - yp - 1];
            output_3[xp][yp] = input_3[xp][yp];
            output_3[xp][img_height - yp - 1] = input_3[xp][img_height - yp - 1];

            sel_r[xp][yp] = output_1[xp][yp] < output_2[xp][yp] && output_1[xp][yp] < output_3[xp][yp];
            sel_g[xp][yp] = output_2[xp][yp] < output_1[xp][yp] && output_2[xp][yp] < output_3[xp][yp];
            sel_b[xp][yp] = output_3[xp][yp] < output_1[xp][yp] && output_2[xp][yp] < output_3[xp][yp];

            sel_r[xp][img_height - yp - 1] = output_1[xp][img_height - yp - 1] < output_2[xp][img_height - yp - 1] && output_1[xp][img_height - yp - 1] < output_3[xp][img_height - yp - 1];
            sel_g[xp][img_height - yp - 1] = output_2[xp][img_height - yp - 1] < output_1[xp][img_height - yp - 1] && output_2[xp][img_height - yp - 1] < output_3[xp][img_height - yp - 1];
            sel_b[xp][img_height - yp - 1] = output_3[xp][img_height - yp - 1] < output_1[xp][img_height - yp - 1] && output_2[xp][img_height - yp - 1] < output_3[xp][img_height - yp - 1];
        }
    }
    for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
        for(int xp = 0; xp < p.r+p.f; xp++){
            output_1[xp][yp] = input_1[xp][yp];
            output_1[img_width - xp - 1][yp] = input_1[img_width - xp - 1][yp];
            output_2[xp][yp] = input_2[xp][yp];
            output_2[img_width - xp - 1][yp] = input_2[img_width - xp - 1][yp];
            output_3[xp][yp] = input_3[xp][yp];
            output_3[img_width - xp - 1][yp] = input_3[img_width - xp - 1][yp];

            sel_r[xp][yp] = output_1[xp][yp] < output_2[xp][yp] && output_1[xp][yp] < output_3[xp][yp];
            sel_g[xp][yp] = output_2[xp][yp] < output_1[xp][yp] && output_2[xp][yp] < output_3[xp][yp];
            sel_b[xp][yp] = output_3[xp][yp] < output_1[xp][yp] && output_2[xp][yp] < output_3[xp][yp];
            sel_r[img_width - xp - 1][yp] = output_1[img_width - xp - 1][yp] < output_2[img_width - xp - 1][yp] && output_1[img_width - xp - 1][yp] < output_3[img_width - xp - 1][yp];
            sel_g[img_width - xp - 1][yp] = output_2[img_width - xp - 1][yp] < output_1[img_width - xp - 1][yp] && output_2[img_width - xp - 1][yp] < output_3[img_width - xp - 1][yp];
            sel_b[img_width - xp - 1][yp] = output_3[img_width - xp - 1][yp] < output_1[img_width - xp - 1][yp] && output_2[img_width - xp - 1][yp] < output_3[img_width - xp - 1][yp];
        }
    }


    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {

            sum_weights = EPSILON;

            // Init output_1 to 0 => TODO: maybe we can do this with calloc
            output_1[xp][yp] = 0.f;
            output_2[xp][yp] = 0.f;
            output_3[xp][yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight_opt1(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;

                    // Add contribution term
                    output_1[xp][yp] += input_1[xq][yq] * wc;
                    output_2[xp][yp] += input_2[xq][yq] * wc;
                    output_3[xp][yp] += input_3[xq][yq] * wc;
                }
            }

            // Normalization step
            output_1[xp][yp] /= sum_weights;
            output_2[xp][yp] /= sum_weights;
            output_3[xp][yp] /= sum_weights;
            sel_r[xp][yp] = output_1[xp][yp] < output_2[xp][yp] && output_1[xp][yp] < output_3[xp][yp];
            sel_g[xp][yp] = output_2[xp][yp] < output_1[xp][yp] && output_2[xp][yp] < output_3[xp][yp];
            sel_b[xp][yp] = output_3[xp][yp] < output_1[xp][yp] && output_2[xp][yp] < output_3[xp][yp];
        }
    }


}

void flt_Basic(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height) {
    scalar wc, wf, w;
    scalar sum_weights;

    buffer gradients;
    if(f != NULL) {
        allocate_buffer(&gradients, img_width, img_height);
        for(int i=0; i<NB_FEATURES;++i) {
            compute_gradient_Basic(gradients[i], f[i], p.r+p.f, img_width, img_height);
        }
    }

    // For edges, just copy in output the input
      for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                out[i][xp][yp] = input[i][xp][yp];
                out[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][xp][img_height - yp - 1] = 0.f;
            }
        }
        for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                out[i][xp][yp] = input[i][xp][yp];
                out[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][img_width - xp - 1][yp] = 0.f;
            }
        }

    }

    // Real computation
    sum_weights = 0;
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {
            
            sum_weights = EPSILON;
            
            for(int i=0;i<3;++i)
                out[i][xp][yp] = 0; 

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    wc = color_weight_opt1(u, var_u, p, xp, yp, xq, yq);

                    if(f != NULL)
                        wf = feature_weight_opt1(f, var_f, gradients, p, xp, yp, xq, yq);
                    else
                        wf = wc;

                    w = fmin(wc, wf);
                    sum_weights += w;

                    for(int i=0;i<3;++i){
                         out[i][xp][yp] += input[i][xq][yq] * w;
                    }

                }
            }

            for(int i=0;i<3;++i){
                out[i][xp][yp] /= sum_weights;

                // ToDo: Fix derivatives => Use formula from paper
                d_out_d_in[i][xp][yp] = 0.f;
            }
        }
    }

    if(f != NULL) {
        free_buffer(&gradients, img_width);
    }
}


void flt_opt1(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height) {
    scalar wc, wf, w;
    scalar sum_weights;

    buffer gradients;
    if(f != NULL) {
        allocate_buffer(&gradients, img_width, img_height);
        for(int i=0; i<NB_FEATURES;++i) {
            compute_gradient_opt1(gradients[i], f[i], p.r+p.f, img_width, img_height);
        }
    }

    // For edges, just copy in output the input
      for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                out[i][xp][yp] = input[i][xp][yp];
                out[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][xp][img_height - yp - 1] = 0.f;
            }
        }
        for (int yp = p.r+p.f; yp < img_height - p.r+p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                out[i][xp][yp] = input[i][xp][yp];
                out[i][img_width - xp - 1][yp] = input[i][img_width - xp - 1][yp];
                d_out_d_in[i][xp][yp] = 0.f;
                d_out_d_in[i][img_width - xp - 1][yp] = 0.f;
            }
        }

    }

    // Real computation
    sum_weights = 0;
    for(int xp = p.r+p.f; xp < img_width-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < img_height-p.r-p.f; ++yp) {
            
            sum_weights = EPSILON;
            
            for(int i=0;i<3;++i)
                out[i][xp][yp] = 0; 

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    wc = color_weight_opt1(u, var_u, p, xp, yp, xq, yq);

                    if(f != NULL)
                        wf = feature_weight_opt1(f, var_f, gradients, p, xp, yp, xq, yq);
                    else
                        wf = wc;

                    w = fmin(wc, wf);
                    sum_weights += w;

                    for(int i=0;i<3;++i){
                         out[i][xp][yp] += input[i][xq][yq] * w;
                    }

                }
            }

            for(int i=0;i<3;++i){
                out[i][xp][yp] /= sum_weights;

                // ToDo: Fix derivatives => Use formula from paper
                d_out_d_in[i][xp][yp] = 0.f;
            }
        }
    }

    if(f != NULL) {
        free_buffer(&gradients, img_width);
    }
}

scalar color_weight_opt1(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar nlmean = nl_means_weights_opt1(u, var_u, p, xp, yp, xq, yq);
    return exp(-fmax(0.f, nlmean));
}

scalar nl_means_weights_opt1(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar distance = 0.f;
    for(int xn = -p.f; xn <= p.f; xn++) {
        for(int yn = -p.f; yn <= p.f; yn++) {
            for(int i=0;i<3;++i) {
                distance += per_pixel_distance_opt1(u[i], var_u[i], p.kc, xp + xn, yp + yn, xq + xn, yq + yn);
            }
        }
    }
    return distance / (scalar)(3*(2*p.f+1)*(2*p.f+1));
}

scalar per_pixel_distance_opt1(channel u, channel var_u, scalar kc, int xp, int yp, int xq, int yq) {
    scalar sqdist = u[xp][yp] - u[xq][yq];
    sqdist *= sqdist;
    scalar var_cancel = var_u[xp][yp] + fmin(var_u[xp][yp], var_u[xq][yq]);
    scalar normalization = EPSILON + kc*kc*(var_u[xp][yp] + var_u[xq][yq]);
    return (sqdist - var_cancel) / normalization;
}


void compute_gradient_Basic(channel gradient, channel u, int d, int img_width, int img_height) {

    for(int x = d; x < img_width-d; ++x) {
        for(int y = d; y < img_height-d; ++y) {
            scalar diffL = u[x][y] - u[x-1][y];
            scalar diffR = u[x][y] - u[x+1][y];
            scalar diffU = u[x][y] - u[x][y-1];
            scalar diffD = u[x][y] - u[x][y+1];

            gradient[x][y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
        }
    }
}

// Optimization possible here since we are only interested in the squared difference
void compute_gradient_opt1(channel gradient, channel u, int d, int img_width, int img_height) {
    scalar diffU_squared;
    scalar diffD_squared;
    for(int x = d; x < img_width-d; ++x) {
        diffD_squared = (u[x][d]-u[x][d-1]);
        diffD_squared *= diffD_squared;
        for(int y = d; y < img_height-d; ++y) {
            scalar diffL = u[x][y] - u[x-1][y];
            scalar diffR = u[x][y] - u[x+1][y];
            diffU_squared = diffD_squared;
            diffD_squared = u[x][y] - u[x][y+1];
            diffD_squared *= diffD_squared;

            gradient[x][y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU_squared, diffD_squared);
        }
    }
}

scalar feature_weight_opt1(channel *f, channel *var_f, channel *gradients, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar df = 0.f;
    for(int j=0; j<NB_FEATURES;++j)
        df = fmax(df, feature_distance_opt1(f[j], var_f[j], gradients[j], p, xp, yp, xq, yq));
    return exp(-df);
}

scalar feature_distance_opt1(channel f, channel var_f, channel gradient, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar sqdist = f[xp][yp] - f[xq][yq];
    sqdist *= sqdist;
    scalar var_cancel = var_f[xp][yp] + fmin(var_f[xp][yp], var_f[xq][yq]);
    scalar normalization = p.kf*p.kf*fmax(p.tau, fmax(var_f[xp][yp], gradient[xp][yp]));
    return (sqdist - var_cancel)/normalization;
}
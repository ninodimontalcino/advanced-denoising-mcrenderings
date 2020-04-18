#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flt.hpp"
#include "memory_mgmt.hpp"




void sure(channel output, buffer c, buffer c_var, buffer cand, buffer cand_d, int img_width, int img_height){

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
                sure += d - v + (v * cand_d[i][x][y]); 

            }
            // Store sure error estimate
            output[x][y] = sure;
        }
    }
}



void flt_buffer_basic(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < img_width; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i][xp][yp] = input[i][xp][yp];
                output[i][xp][img_height - yp - 1] = input[i][xp][img_height - yp - 1];
            }
        }
        for (int yp = 0; yp < img_height; yp++){
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
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
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


void flt_channel_basic(channel output, channel input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height){

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < img_width; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            output[xp][yp] = input[xp][yp];
            output[xp][img_height - yp - 1] = input[xp][img_height - yp - 1];
        }
    }
    for (int yp = 0; yp < img_height; yp++){
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
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    sum_weights += wc;

                    // Add contribution term
                    output[xp][yp] += input[xq][yq] * wc;
                }
            }

            // Normalization step
            for(int i=0;i<3;++i)
                output[xp][yp] /= (sum_weights + EPSILON);
        }
    }


}


void flt(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height) {
    scalar wc, wf, w;
    scalar sum_weights;

    buffer gradients;
    if(f != NULL) {
        allocate_buffer(&gradients, img_width, img_height);
        for(int i=0; i<NB_FEATURES;++i) {
            compute_gradient(gradients[i], f[i], p.r+p.f, img_width, img_height);
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
        for (int yp = 0; yp < img_height; yp++){
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
            
            sum_weights = 0.f;
            
            for(int i=0;i<3;++i)
                out[i][xp][yp] = 0; 
                for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                    for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                        wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                        if(f != NULL)
                            wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
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
                out[i][xp][yp] /= (sum_weights + EPSILON);

                // ToDo: Fix derivatives => Use formula from paper
                d_out_d_in[i][xp][yp] = 0.f;
            }
        }
    }

    if(f != NULL) {
        free_buffer(&gradients, img_width);
    }
}

scalar color_weight(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar nlmean = nl_means_weights(u, var_u, p, xp, yp, xq, yq);
    return exp(-fmax(0.f, nlmean));
}

scalar nl_means_weights(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar distance = 0.f;
    for(int xn = -p.f; xn <= p.f; xn++) {
        for(int yn = -p.f; yn <= p.f; yn++) {
            for(int i=0;i<3;++i) {
                distance += per_pixel_distance(u[i], var_u[i], p.kc, xp + xn, yp + yn, xq + xn, yq + yn);
            }
        }
    }
    return distance / (scalar)(3*(2*p.f+1)*(2*p.f+1));
}

scalar per_pixel_distance(channel u, channel var_u, scalar kc, int xp, int yp, int xq, int yq) {
    scalar sqdist = u[xp][yp] - u[xq][yq];
    sqdist *= sqdist;
    scalar var_cancel = var_u[xp][yp] + fmin(var_u[xp][yp], var_u[xq][yq]);
    scalar normalization = EPSILON + kc*kc*(var_u[xp][yp] + var_u[xq][yq]);
    return (sqdist - var_cancel) / normalization;
}

void compute_gradient(channel gradient, channel u, int d, int img_width, int img_height) {
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

scalar feature_weight(channel *f, channel *var_f, channel *gradients, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar df = 0.f;
    for(int j=0; j<NB_FEATURES;++j)
        df = fmax(df, feature_distance(f[j], var_f[j], gradients[j], p, xp, yp, xq, yq));
    return exp(-df);
}

scalar feature_distance(channel f, channel var_f, channel gradient, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar sqdist = f[xp][yp] - f[xq][yq];
    sqdist *= sqdist;
    scalar var_cancel = var_f[xp][yp] + fmin(var_f[xp][yp], var_f[xq][yq]);
    scalar normalization = p.kf*p.kf*fmax(p.tau, fmax(var_f[xp][yp], gradient[xp][yp]));
    return (sqdist - var_cancel)/normalization;
}
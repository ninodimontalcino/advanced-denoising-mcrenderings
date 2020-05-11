#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flt.hpp"
#include "memory_mgmt.hpp"



void sure(scalar* output, scalar* c, scalar* c_var, scalar* cand, scalar* cand_d, int W, int H){

    int WH = W*H;

    scalar d, v;
    
    for (int x = 0; x < W; x++){
        for (int y = 0; y < H; y++){

            scalar sure = 0.f;

            // Sum over color channels
            for (int i = 0; i < 3; i++){    

                // Calculate terms
                d = cand[i * WH + x * W + y] - c[i * WH + x * W + y];
                d *= d;
                v = c_var[i * WH + x * W + y];
                v *= v;

                // Summing up
                sure += d - v + (2 * v * cand_d[i * WH + x * W + y]); 

            }
            // Store sure error estimate
            output[x * W + y] = sure;
        }
    }
}



void flt_buffer_basic(scalar* output, scalar* input, scalar* u, scalar* var_u, Flt_parameters p, int W, int H){

    int WH = W*H;

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                output[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                output[i * WH + xp * W + H - yp - 1] = input[i * WH + xp * W + H - yp - 1];
            }
        }
        for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                output[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                output[i * WH + (W - xp - 1) * W + yp] = input[i * WH + (W - xp - 1) * W + yp];
            }
        }
    }

    
    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < W - p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < H - p.r-p.f; ++yp) {

            sum_weights = 0.f;

            // Init output to 0 => TODO: maybe we can do this with calloc
            for(int i = 0; i < 3; ++i)
                output[i * WH + xp * W + yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq, W, H);
                    sum_weights += wc;

                    // Add contribution term
                    for(int i=0;i<3;++i)
                        output[i * WH + xp * W + yp] += input[i * WH + xq * W + yq] * wc;
                }
            }

            // Normalization step
            for(int i=0;i<3;++i)
                output[i * WH + xp * W + yp] /= (sum_weights + EPSILON);
        }
    }
    
}


void flt_channel_basic(scalar* output, scalar* input, scalar* u, scalar* var_u, Flt_parameters p, int W, int H){

    int WH = W*H;

    scalar sum_weights, wc;

    // Handling Border Cases (border section)
    for (int xp = 0; xp < W; xp++){
        for(int yp = 0; yp < p.r+p.f; yp++){
            output[xp * W + yp] = input[xp * W + yp];
            output[xp * W + H - yp - 1] = input[xp * W + H - yp - 1];
        }
    }
    for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
        for(int xp = 0; xp < p.r+p.f; xp++){
            output[xp * W + yp] = input[xp * W + yp];
            output[(W - xp - 1) * W + yp] = input[(W - xp - 1) * W + yp];
        }
    }
       

    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < W-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < H-p.r-p.f; ++yp) {

            sum_weights = 0.f;

            // Init output to 0 => TODO: maybe we can do this with calloc
            output[xp * W + yp] = 0.f;

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    // Compute color Weight
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq, W, H);
                    sum_weights += wc;

                    // Add contribution term
                    output[xp * W + yp] += input[xq * W + yq] * wc;
                }
            }

            // Normalization step
            output[xp * W + yp] /= (sum_weights + EPSILON);
        }
    }


}


void flt(scalar* out, scalar* d_out_d_in, scalar* input, scalar* u, scalar* var_u, scalar* f, scalar* var_f, Flt_parameters p, int W, int H) {
    
    int WH = W*H;
    
    scalar wc, wf, w;
    scalar sum_weights;

    scalar* gradients;
    if(f != NULL) {
        allocate_buffer(&gradients, W, H);
        compute_gradient(gradients, f, p.r+p.f, W, H);
    }

    for (int i = 0; i < 3; i++){
        for (int xp = 0; xp < W; xp++){
            for(int yp = 0; yp < p.r+p.f; yp++){
                out[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                out[i * WH + xp * W + H - yp - 1] = input[i * WH + xp * W + H - yp - 1];
                d_out_d_in[i * WH + xp * W + yp] = 0.f;
                d_out_d_in[i * WH + xp * W + H - yp - 1] = 0.f;
            }
        }
        for (int yp = p.r+p.f ; yp < H - p.r - p.f; yp++){
            for(int xp = 0; xp < p.r+p.f; xp++){
                out[i * WH + xp * W + yp] = input[i * WH + xp * W + yp];
                out[i * WH + (W - xp - 1) * W + yp] = input[i * WH + (W - xp - 1) * W + yp];
                d_out_d_in[i * WH + xp * W + yp] = 0.f;
                d_out_d_in[i * WH + (W - xp - 1) * W + yp] = 0.f;
            }
        }
    }

    // Real computation
    sum_weights = 0;
    for(int xp = p.r+p.f; xp < W-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < H-p.r-p.f; ++yp) {
            
            sum_weights = 0.f;
            
            for(int i=0;i<3;++i)
                out[i * WH + xp * W + yp] = 0; 

            for(int xq = xp-p.r; xq <= xp+p.r; xq++) {
                for(int yq = yp-p.r; yq <= yp+p.r; yq++) {
                    
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq, W, H);

                    if(f != NULL)
                        wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq, W, H);
                    else
                        wf = wc;

                    w = fmin(wc, wf);
                    sum_weights += w;

                    for(int i=0;i<3;++i){
                         out[i * WH + xp * W + yp] += input[i * WH + xq * W + yq] * w;
                    }

                }
            }

            for(int i=0;i<3;++i){
                out[i * WH + xp * W + yp] /= (sum_weights + EPSILON);

                // ToDo: Fix derivatives => Use formula from paper
                d_out_d_in[i * WH + xp * W + yp] = 0.f;
            }
        }
    }

    if(f != NULL) {
        free_buffer(&gradients);
    }
}

scalar color_weight(scalar* u, scalar* var_u, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H) {
    scalar nlmean = nl_means_weights(u, var_u, p, xp, yp, xq, yq, W, H);
    return exp(-fmax(0.f, nlmean));
}

scalar nl_means_weights(scalar* u, scalar* var_u, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H) {

    scalar distance = 0.f;
    for(int xn = -p.f; xn <= p.f; xn++) {
        for(int yn = -p.f; yn <= p.f; yn++) {
            for(int i=0;i<3;++i) {
                distance += per_pixel_distance(u + i * W * H, var_u + i * W * H, p.kc, xp + xn, yp + yn, xq + xn, yq + yn, W, H);
            }
        }
    }
    return distance / (scalar)(3*(2*p.f+1)*(2*p.f+1));
}

scalar per_pixel_distance(scalar* u, scalar* var_u, scalar kc, int xp, int yp, int xq, int yq, int W, int H) {
    scalar sqdist = u[xp * W + yp] - u[xq * W + yq];
    sqdist *= sqdist;
    scalar var_cancel = var_u[xp * W + yp] + fmin(var_u[xp * W + yp], var_u[xq * W + yq]);
    scalar normalization = EPSILON + kc*kc*(var_u[xp * W + yp] + var_u[xq * W + yq]);
    return (sqdist - var_cancel) / normalization;
}

void compute_gradient(scalar* gradients, scalar* u, int d, int W, int H) {
    
    int WH = W*H;

    for(int i=0; i<NB_FEATURES;++i) {
        for(int x = d; x < W -d; ++x) {
            for(int y = d; y < H - d; ++y) {
                scalar diffL = u[i * WH + x * W + y] - u[i * WH + (x-1) * W + y];
                scalar diffR = u[i * WH + x * W + y] - u[i * WH + (x+1) * W + y];
                scalar diffU = u[i * WH + x * W + y] - u[i * WH + x * W + y - 1];
                scalar diffD = u[i * WH + x * W + y] - u[i * WH + x * W + y + 1];

                gradients[i * WH + x * W + y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
            }
        } 
    }
}

scalar feature_weight(scalar* f, scalar* var_f, scalar* gradients, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H) {
    
    int WH = W * H;
    
    scalar df = 0.f;
    for(int j=0; j<NB_FEATURES;++j)
        df = fmax(df, feature_distance(f + j*WH, var_f + j*WH, gradients + j*WH, p, xp, yp, xq, yq, W, H));
    return exp(-df);
}

scalar feature_distance(scalar* f, scalar* var_f, scalar* gradient, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H) {
    scalar sqdist = f[xp * W + yp] - f[xq * W + yq];
    sqdist *= sqdist;
    scalar var_cancel = var_f[xp * W + yp] + fmin(var_f[xp * W + yp], var_f[xq * W + yq]);
    scalar normalization = p.kf*p.kf*fmax(p.tau, fmax(var_f[xp * W + yp], gradient[xp * W + yp]));
    return (sqdist - var_cancel)/normalization;
}
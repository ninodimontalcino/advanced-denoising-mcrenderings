#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flt.hpp"



void sure(channel output, buffer c, buffer c_var, buffer cand, buffer cand_d, int img_width, int img_height){

    double d, v;
    
    for (int x = 0; x < img_width; x++){
        for (int y = 0; y < img_height; y++){

            double sure = 0;

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



void flt_buffer_basic(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p){

    double sum_weights, wc;

    // Handling Border Cases (border section)
    for(int i = 0; i < 3; i++) {
        for(int xp = 0; xp < p.r+p.f; ++xp) {
            for(int yp = 0; yp < p.r+p.f; ++yp)
                output[i][xp][yp] = input[i][xp][yp];
            for(int yp = IMG_H-p.r-p.f; yp < IMG_H; ++yp)
                output[i][xp][yp] = input[i][xp][yp];
        }
        for(int xp = IMG_W-p.r-p.f; xp < IMG_W; ++xp) {
            for(int yp=0;yp<p.r+p.f;++yp)
                output[i][xp][yp] = input[i][xp][yp];
            for(int yp = IMG_H-p.r-p.f; yp < IMG_H; ++yp)
                output[i][xp][yp] = input[i][xp][yp];
        }
        
    }

    
    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < IMG_W-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < IMG_H-p.r-p.f; ++yp) {

            sum_weights = 0;

            // Init output to 0 => TODO: maybe we can do this with calloc
            for(int i=0;i<3;++i)
                output[i][xp][yp] = 0;

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
                output[i][xp][yp] /= sum_weights;
        }
    }
    


}


void flt_channel_basic(channel output, channel input, buffer u, buffer var_u, Flt_parameters p){

    double sum_weights, wc;

    // Handling Border Cases (border section)
    for(int i = 0; i < 3; ++i) {
        for(int xp = 0; xp < p.r+p.f; ++xp) {
            for(int yp = 0; yp < p.r+p.f; ++yp)
                output[xp][yp] = input[xp][yp];
            for(int yp = IMG_H-p.r-p.f; yp < IMG_H; ++yp)
                output[xp][yp] = input[xp][yp];
        }
        for(int xp = IMG_W-p.r-p.f; xp < IMG_W; ++xp) {
            for(int yp=0;yp<p.r+p.f;++yp)
                output[xp][yp] = input[xp][yp];
            for(int yp = IMG_H-p.r-p.f; yp < IMG_H; ++yp)
                output[xp][yp] = input[xp][yp];
        }
    }

    // General Pre-Filtering
    for(int xp = p.r+p.f; xp < IMG_W-p.r-p.f; ++xp) {
        for(int yp = p.r+p.f; yp < IMG_H-p.r-p.f; ++yp) {

            sum_weights = 0;

            // Init output to 0 => TODO: maybe we can do this with calloc
            output[xp][yp] = 0;

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
                output[xp][yp] /= sum_weights;
        }
    }


}


void flt(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p) {
    double wc, wf, w;
    double sum_weights;

    channel* gradients;
    if(f != NULL) {
        gradients = (channel *)malloc(NB_FEATURES*sizeof(void*));
        for(int i=0; i<NB_FEATURES;++i) {
            gradients[i] = (channel)malloc(IMG_W*sizeof(void *));
            for(int j=0; j<IMG_W;++j)
                gradients[i][j] = (scalar*)malloc(IMG_H*sizeof(scalar));
            compute_gradient(gradients[i], f[i], p.r+p.f);
        }
    }

    // For edges, just copy in output the input
    for(int i=0;i<3;++i) {
        for(int xp=0;xp<p.r+p.f;++xp) {
            for(int yp=0;yp<p.r+p.f;++yp)
                out[i][xp][yp] = input[i][xp][yp];
            for(int yp=IMG_H-p.r-p.f;yp<IMG_H;++yp)
                out[i][xp][yp] = input[i][xp][yp];
        }
        for(int xp=IMG_W-p.r-p.f;xp<IMG_W;++xp) {
            for(int yp=0;yp<p.r+p.f;++yp)
                out[i][xp][yp] = input[i][xp][yp];
            for(int yp=IMG_H-p.r-p.f;yp<IMG_H;++yp)
                out[i][xp][yp] = input[i][xp][yp];
        }
    }

    // Real computation
    sum_weights = 0;
    for(int xp=p.r+p.f;xp<IMG_W-p.r-p.f;++xp) {
        for(int yp=p.r+p.f;yp<IMG_H-p.r-p.f;++yp) {
            sum_weights = 0;
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
                    for(int i=0;i<3;++i)
                        out[i][xp][yp] += input[i][xq][yq] * w;
                }
            }
            for(int i=0;i<3;++i)
                out[i][xp][yp] /= sum_weights;
        }
    }

    if(f != NULL) {
        for(int i=0; i<NB_FEATURES;++i) {
            for(int j=0; j<IMG_W;++j)
                free(gradients[i][j]);
            free(gradients[i]);
        }
        free(gradients);
    }
}

scalar color_weight(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar nlmean = nl_means_weights(u, var_u, p, xp, yp, xq, yq);
    return exp(-fmax(0, nlmean));
}

scalar nl_means_weights(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar distance = 0;
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

void compute_gradient(channel gradient, channel u, int d) {
    for(int x=d; x<IMG_W-d; ++x) {
        for(int y=d; y<IMG_H-d; ++y) {
            double diffL = u[x][y] - u[x-1][y];
            double diffR = u[x][y] - u[x+1][y];
            double diffU = u[x][y] - u[x][y-1];
            double diffD = u[x][y] - u[x][y+1];

            gradient[x][y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
        }
    }
}

scalar feature_weight(channel *f, channel *var_f, channel *gradients, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar df = 0;
    for(int j=0; j<NB_FEATURES;++j)
        df = fmax(df, feature_distance(f[j], var_f[j], gradients[j], p, xp, yp, xq, yq));
    return exp(-df);
}

scalar feature_distance(channel f, channel var_f, channel gradient, Flt_parameters p, int xp, int yp, int xq, int yq) {
    scalar sqdist = f[xp][yp] - f[xq][yq];
    sqdist *= sqdist;
    double var_cancel = var_f[xp][yp] + fmin(var_f[xp][yp], var_f[xq][yq]);
    double normalization = p.kf*p.kf*fmax(p.tau, fmax(var_f[xp][yp], gradient[xp][yp]));
    return (sqdist - var_cancel)/normalization;
}
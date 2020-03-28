#include <stdlib.h>
#include <math.h>
#include "flt.h"

void flt(buffer *out, buffer *d_out_d_in, buffer *input, buffer *u, buffer *var_u, buffer *f, buffer *var_f, Flt_parameters p) {
    double wc, wf;
    double sum_weights;

    buffer *gradients[NB_FEATURES][IMG_SIZE][IMG_SIZE];
    for(int i=0; i<NB_FEATURES;++i) {
        compute_gradient(gradients[i], f[i]);
    }

    sum_weights = 0;
    for(int xp=0;xp<IMG_SIZE;++xp) {
        for(int yp=0;yp<IMG_SIZE;++yp) {
            for(int xq = MIN(xp-p.r, 0); xq <= MAX(xp+p.r, IMG_SIZE-1); xq++) {
                for(int yq = MIN(yp-p.r, 0); yq <= MAX(yp+p.r, IMG_SIZE-1); yq++) {
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
                    wf = feature_weight(f, var_f, gradients, p, xp, yp, xq, yq);
                    // TODO complete
                }
            }
        }
    }
}

double color_weight(buffer *u, buffer *var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    double nlmean = nl_means_weights(u, var_u, p, xp, yp, xq, yq);
    return exp(-fmax(0, nlmean));
}

double nl_means_weights(buffer *u, buffer *var_u, Flt_parameters p, int xp, int yp, int xq, int yq) {
    double distance = 0;
    for(int xn = -p.r; xn <= p.f; xn++) {
        for(int yn = -p.r; yn <= p.f; yn++) {
            for(int i=0;i<3;++i) {
                // TODO check boundaries
                distance += per_pixel_distance(u[i], var_u[i], p.kc, xp + xn, yp + yn, xq + xn, yq + yn);
            }
        }
    }
    return distance / (double)(3*(2*p.f+1)*(2*p.f+1));
}

double per_pixel_distance(buffer u, buffer var_u, double kc, int xp, int yp, int xq, int yq) {
    double sqdist = u[xp][yp] - u[xq][yq];
    sqdist *= sqdist;
    double var_cancel = var_u[xp][yp] + fmin(var_u[xp][yp], var_u[xq][yq]);
    double normalization = EPSILON + kc*kc*(var_u[xp][yp] + var_u[xq][yq]);
    return (sqdist - var_cancel) / normalization;
}

void compute_gradient(buffer gradient, buffer u) {
    for(int x=0; x<IMG_SIZE; ++x) {
        for(int y=0; y<IMG_SIZE; ++y) {
            //TODO check bounds
            double diffL = u[x][y] - u[x-1][y];
            double diffR = u[x][y] - u[x+1][y];
            double diffU = u[x][y] - u[x][y-1];
            double diffD = u[x][y] - u[x][y+1];

            gradient[x][y] = fmin(diffL*diffL, diffR*diffR) + fmin(diffU*diffU, diffD*diffD);
        }
    }
}

double feature_weight(buffer *f, buffer *var_f, buffer *gradients, Flt_parameters p, int xp, int yp, int xq, int yq) {
    double df = 0;
    for(int j=0; j<NB_FEATURES;++j)
        df = fmax(df, feature_distance(f[j], var_f[j], gradients[j], p, xp, yp, xq, yq));
    return exp(-df);
}

double feature_distance(buffer f, buffer var_f, buffer gradient, Flt_parameters p, int xp, int yp, int xq, int yq) {
    double sqdist = f[xp][yp] - f[xq][yq];
    sqdist *= sqdist;
    double var_cancel = var_f[xp][yp] + fmin(var_f[xp][yp], var_f[xq][yq]);
    double normalization = p.kf*p.kf*fmax(p.tau, fmax(var_f[xp][yp], gradient[xp][yp]));
    return (sqdist - var_cancel)/normalization;
}
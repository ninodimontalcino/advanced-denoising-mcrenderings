#include <stdlib.h>
#include <math.h>
#include "flt.h"

void flt(buffer *out, buffer *d_out_d_in, buffer *input, buffer *u, buffer *var_u, buffer *f, buffer *var_f, Flt_parameters p) {
    double wc;
    double sum_weights;

    sum_weights = 0;
    for(int xp=0;xp<IMG_SIZE;++xp) {
        for(int yp=0;yp<IMG_SIZE;++yp) {
            for(int xq = MIN(xp-p.r, 0); xq <= MAX(xp+p.r, IMG_SIZE-1); xq++) {
                for(int yq = MIN(yp-p.r, 0); yq <= MAX(yp+p.r, IMG_SIZE-1); yq++) {
                    wc = color_weight(u, var_u, p, xp, yp, xq, yq);
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
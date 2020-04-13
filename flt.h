#ifndef FLT_H
#define FLT_H

typedef float scalar;
typedef scalar** channel;
typedef scalar*** buffer;
#define IMG_W 800
#define IMG_H 600

#define EPSILON 0.01
#define NB_FEATURES 3

typedef struct
{
    scalar kc; //Sensitivity to color
    scalar kf; //Sensitivity to features
    scalar tau; //Threshold
    int f; //Patch radius
    int r; //Window radius
} Flt_parameters;

/**
 * flt performs the filter function (convolution).
 * 
 * Parameters:
 *  - input: input buffer
 *  - u: color buffer (3 colors)
 *  - var_u: variance of u
 *  - f: feature buffers. May be NULL if no feature buffer wanted.
 *  - var_f: variance of feature buffers
 *  - p: flt parameters
 * 
 * Output:
 *  - out: filtered buffer
 *  - d_out_d_in: derivative needed for SURE estimator.
*/
void flt(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, channel *f, channel *var_f, Flt_parameters p);

scalar per_pixel_distance(channel u, channel var_u, scalar kc, int xp, int yp, int xq, int yq);
scalar nl_means_weights(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq);
scalar color_weight(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq);

void compute_gradient(channel gradient, channel u, int d);
scalar feature_distance(channel f, channel var_f, channel gradient, Flt_parameters p, int xp, int yp, int xq, int yq);
scalar feature_weight(channel *f, channel *var_f, channel *gradients, Flt_parameters p, int xp, int yp, int xq, int yq);


#endif //FLT_H
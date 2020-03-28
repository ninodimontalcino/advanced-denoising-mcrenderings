#ifndef FLT_H
#define FLT_H

#define IMG_SIZE 512
#define EPSILON 0.01
#define NB_FEATURES 3

typedef double** buffer;
typedef struct
{
    double kc; //Sensitivity to color
    double kf; //Sensitivity to features
    double tau; //Threshold
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
 *  - f: feature buffers
 *  - var_f: variance of feature buffers
 *  - p: flt parameters
 * 
 * Output:
 *  - out: filtered buffer
 *  - d_out_d_in: derivative needed for SURE estimator.
*/
void flt(buffer *out, buffer *d_out_d_in, buffer *input, buffer *u, buffer *var_u, buffer *f, buffer *var_f, Flt_parameters p);

//Note: those macro are only used for integers
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

double per_pixel_distance(buffer u, buffer var_u, double kc, int xp, int yp, int xq, int yq);
double nl_means_weights(buffer *u, buffer *var_u, Flt_parameters p, int xp, int yp, int xq, int yq);
double color_weight(buffer *u, buffer *var_u, Flt_parameters p, int xp, int yp, int xq, int yq);

void compute_gradient(buffer gradient, buffer u);
double feature_distance(buffer f, buffer var_f, buffer gradient, Flt_parameters p, int xp, int yp, int xq, int yq);
double feature_weight(buffer *f, buffer *var_f, buffer *gradients, Flt_parameters p, int xp, int yp, int xq, int yq);

#endif //FLT_H
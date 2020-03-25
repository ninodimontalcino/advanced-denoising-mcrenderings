#ifndef FLT_H
#define FLT_H

typedef double** buffer;
typedef struct
{
    double kc; //Sensitivity to color
    double kf; //Sensitivity to features
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
void flt(buffer out, buffer d_out_d_in, buffer input, buffer *u, buffer *var_u, buffer *f, buffer *var_f, Flt_parameters p);

#endif //FLT_H
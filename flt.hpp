#ifndef FLT_H
#define FLT_H

typedef float scalar;
typedef scalar** channel;
typedef scalar*** buffer;

#define EPSILON 0.000000000001
#define NB_FEATURES 3

typedef struct
{
    scalar kc; //Sensitivity to color
    scalar kf; //Sensitivity to features
    scalar tau; //Threshold
    int f; //Patch radius
    int r; //Window radius
} Flt_parameters;


/* -------------------------------------------------------
    SURE Error Estimator

    Parameters:
        - output (channel):     buffer containing SURE error estimate
        - c (buffer):           color buffer
        - c_var (buffer):       variance of c
        - cand (buffer):        candidate filter
        - cand_d (buffer):      derivative of candidate filter
        - W (int):      image width
        - H (int):     image height

*/
void sure(scalar*  output, scalar* c, scalar* c_var, scalar* cand, scalar* cand_d, int W, int H);

/* -------------------------------------------------------
    Basic Filtering (Buffer)

    Parameters:
        - output (buffer):  buffer for filtered image (OUTPUT)
        - input (buffer):   input buffer => to be filtered
        - u (buffer):       color buffer
        - var_u (buffer):   variance of u
        - p (struct):       pre-filtering parameters 

    Returns:
        - output (buffer):  filtered image in buffer output
   
*/
void flt_buffer_basic(scalar* output, scalar* input, scalar* u, scalar* var_u, Flt_parameters p, int W, int H);


/* -------------------------------------------------------
    Basic Filtering (Channel)

    Parameters:
        - output (channel): channel for filtered image (OUTPUT)
        - input (channel):  input channel => to be filtered
        - u (buffer):       color buffer
        - var_u (buffer):   variance of u
        - p (struct):       pre-filtering parameters 

    Returns:
        - output (buffer):  filtered image in buffer output
   
*/
void flt_channel_basic(scalar*  output, scalar* input, scalar* u, scalar* var_u, Flt_parameters p, int W, int H);

/* -------------------------------------------------------
    Main Filtering (color and feature input)

    Parameters:
        - out (buffer)          buffer for filtered buffer (OUTPUT POINTER)
        - d_out_d_in (buffer)   buffer for derivative needed for SURE ESIMTATOR  (OUTPUT POINTER)
        - input (buffer)        input buffer => to be filtered
        - u (buffer)            color buffer (3 color channels)
        - var_u (buffer)        variance of u
        - f (buffer)            feature buffers. May be NULL if no feature buffer wanted.
        - var_f (buffer)        variance of feature buffers
        - p (struct)            flt parameters

    Returns:
        - out (buffer)           filtered buffer
        - d_out_d_in (buffer)    Corresponding derrivative "d_out_d_in" 
        
*/        
void flt(scalar* out, scalar* d_out_d_in, scalar* input, scalar* u, scalar* var_u, scalar* f, scalar* var_f, Flt_parameters p, int W, int H);

scalar per_pixel_distance(scalar*  u, scalar*  var_u, scalar kc, int xp, int yp, int xq, int yq, int W, int H);
scalar nl_means_weights(scalar* u, scalar* var_u, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H);
scalar color_weight(scalar* u, scalar* var_u, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H);

void compute_gradient(scalar*  gradient, scalar* u, int d, int W, int H);
scalar feature_distance(scalar*  f, scalar*  var_f, scalar*  gradient, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H);
scalar feature_weight(scalar* f, scalar*  var_f, scalar* gradients, Flt_parameters p, int xp, int yp, int xq, int yq, int W, int H);

#endif //FLT_H
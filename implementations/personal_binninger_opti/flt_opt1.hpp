#ifndef FLT_H_OPT1
#define FLT_H_OPT1

#include "../../flt.hpp"

// typedef float scalar;
// typedef scalar** channel;
// typedef scalar*** buffer;

// #define EPSILON 0.0000000001
// #define NB_FEATURES 3

// typedef struct
// {
//     scalar kc; //Sensitivity to color
//     scalar kf; //Sensitivity to features
//     scalar tau; //Threshold
//     int f; //Patch radius
//     int r; //Window radius
// } Flt_parameters;


/* -------------------------------------------------------
    SURE Error Estimator

    Parameters:
        - output (channel):     buffer containing SURE error estimate
        - c (buffer):           color buffer
        - c_var (buffer):       variance of c
        - cand (buffer):        candidate filter
        - cand_d (buffer):      derivative of candidate filter
        - img_width (int):      image width
        - img_height (int):     image height

*/
void sure_Basic(channel output, buffer c, buffer c_var, buffer cand, buffer cand_d, int img_width, int img_height);
void sure_opt1(channel sure_r, channel sure_g, channel sure_b, buffer c, buffer c_var, buffer r, buffer d_r, buffer g, buffer d_g, buffer b, buffer d_b, int img_width, int img_height);

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
void flt_buffer_Basic(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height);
void flt_buffer_opt1(buffer f_filtered, buffer f_var_filtered, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height);


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
void flt_channel_Basic(channel output, channel input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height);
void flt_channel_opt1(channel output_1, channel input_1, channel output_2, channel input_2, channel output_3, channel input_3, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height);
void flt_channel_opt1_sel(channel output_1, channel input_1, channel output_2, channel input_2, channel output_3, channel input_3, channel sel_r, channel sel_g, channel sel_b, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height);

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
void flt_Basic(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height);
void flt_opt1(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int img_width, int img_height);

scalar per_pixel_distance_Basic(channel u, channel var_u, scalar kc, int xp, int yp, int xq, int yq);
scalar per_pixel_distance_opt1(channel u, channel var_u, scalar kc, int xp, int yp, int xq, int yq);

scalar nl_means_weights_opt1(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq);
scalar color_weight_opt1(buffer u, buffer var_u, Flt_parameters p, int xp, int yp, int xq, int yq);

void compute_gradient_Basic(channel gradient, channel u, int d, int img_width, int img_height);
void compute_gradient_opt1(channel gradient, channel u, int d, int img_width, int img_height);

scalar feature_distance_Basic(channel f, channel var_f, channel gradient, Flt_parameters p, int xp, int yp, int xq, int yq);
scalar feature_weight_Basic(channel *f, channel *var_f, channel *gradients, Flt_parameters p, int xp, int yp, int xq, int yq);


#endif //FLT_H_OPT1
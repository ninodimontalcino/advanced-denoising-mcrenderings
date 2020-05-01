#ifndef FLT1_H
#define FLT1_H
#include "flt.hpp"

typedef scalar**** bufferweight;
typedef scalar ***** bufferweightset;

/* -------------------------------------------------------
    Basic Filtering (Buffer)

    Parameters:
        - output (buffer):  buffer for filtered image (OUTPUT)
        - input (buffer):   input buffer => to be filtered
        - u (buffer):       color buffer
        - var_u (buffer):   variance of u
        - allparams (array of struct):      pre-filtering parameters 
        - config (int):                     number of struct in allparams
        - weights (bufferweightset):        buffer with all weights to be precomputed

    Returns:
        - output (buffer):  filtered image in buffer output
   
*/
void precompute_colors_pref(bufferweight weightpref, buffer u, buffer var_u, int img_width, int img_height, Flt_parameters p);
void flt_buffer_opcount(buffer output, buffer input, buffer u, buffer var_u, Flt_parameters p, int img_width, int img_height, bufferweight weights);
void flt_opcount(buffer out, buffer d_out_d_in, buffer input, buffer u, buffer var_u, buffer f, buffer var_f, Flt_parameters p, int config, int img_width, int img_height, bufferweightset weights);
void flt_channel_opcount(channel output, channel input, buffer u, buffer var_u, Flt_parameters p, int config, int img_width, int img_height, bufferweightset weights);
scalar access_weight(bufferweightset weights, int xp, int yp, int xq, int yq, int config);
void precompute_color_weights(bufferweightset allweights, buffer u, buffer var_u, int img_width, int img_height, Flt_parameters* all_params, int n_params);
void precompute_weights(bufferweightset allweights, scalar* allsums, buffer u, buffer var_u,  buffer f, buffer var_f, int img_width, int img_height, Flt_parameters* all_params);

#endif //FLT1_H
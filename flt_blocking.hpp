#ifndef FLT_3_H
#define FLT_3_H

#include "flt.hpp"


void sure_all_blocking(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int img_width, int img_height);
void filtering_basic_blocking(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int img_width, int img_height);
void feature_prefiltering_blocking(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering_blocking(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering_all_blocking(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int img_width, int img_height);
#endif //FLT_2_H
#ifndef FLT_RESTRUCTURE_H
#define FLT_RESTRUCTURE_H

#include "flt.hpp"


void sure_all(scalar* sure, scalar* c, scalar* c_var, scalar* cand_r, scalar* cand_g, scalar* cand_b, int img_width, int img_height);
void filtering_basic(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int img_width, int img_height);
void feature_prefiltering(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering(scalar* output, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering_all(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int img_width, int img_height);
void candidate_filtering_all2(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int W, int H);

#endif //FLT_RESTRUCTURE_H
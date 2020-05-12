#ifndef FLT_RESTRUCTURE_H
#define FLT_RESTRUCTURE_H

#include "flt.hpp"


// ---------------------------------------------
// Default Functions with scalar replacement
// ---------------------------------------------

// To Be Done!!!! => Necessary for step-wise optimization analysis (just remove all scalar replacment steps from default functions)

// ---------------------------------------------
// Default Functions (with scalar replacment)
// ---------------------------------------------

void sure_all(scalar* sure, scalar* c, scalar* c_var, scalar* cand_r, scalar* cand_g, scalar* cand_b, int img_width, int img_height);
void filtering_basic(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int img_width, int img_height);
void feature_prefiltering(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering(scalar* output, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering_all(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int img_width, int img_height);

// ---------------------------------------------
// Functions with ILP
// ---------------------------------------------

void sure_all_ILP(scalar* sure, scalar* c, scalar* c_var, scalar* cand_r, scalar* cand_g, scalar* cand_b, int img_width, int img_height);
void filtering_basic_ILP(scalar* output, scalar* input, scalar* c, scalar* c_var, Flt_parameters p, int img_width, int img_height);
void feature_prefiltering_ILP(scalar* output, scalar* output_var, scalar* features, scalar* features_var, Flt_parameters p, int img_width, int img_height);
void candidate_filtering_all_ILP(scalar* output_r, scalar* output_g, scalar* output_b, scalar* color, scalar* color_var, scalar* features, scalar* features_var, Flt_parameters* p, int img_width, int img_height);

#endif //FLT_RESTRUCTURE_H
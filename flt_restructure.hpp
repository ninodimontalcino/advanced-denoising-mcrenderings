#ifndef FLT_2_H
#define FLT_2_H

#include "flt.hpp"


// TYPE DEFINITION FOR BLOCKING
#define LT 0
#define LB 1
#define RT 2
#define RB 3
#define LL  4
#define RR  5
#define TT  6
#define BB  7
#define II  8



// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// OPTIMIZED VERSION (without scalar replacement)
// ...

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// OPTIMIZED VERSION (with scalar replacement)
void sure_all(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H);
void filtering_basic(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);  // Specific for k_f = Inf
void feature_prefiltering(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_THIRD(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_all(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int W, int H);

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// UNROLLED VERSION for increased ILP
void sure_all_ILP(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H);
void filtering_basic_f3_ILP(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);   // Unrolled for specific f=3 and k_f = Inf
void filtering_basic_f1_ILP(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);   // Unrolled for specific f=1 and k_f = Inf
void feature_prefiltering_ILP(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);                      // Unrolled for specific f=3 and k_f = Inf
void candidate_filtering_FIRST_ILP(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);    // Unrolled for specific for f=1
void candidate_filtering_SECOND_ILP(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);   // Unrolled for specific for f=3
void candidate_filtering_THIRD_ILP(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);    // Specific Function case k_c = Inf
void candidate_filtering_all_ILP(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int W, int H);
void candidate_filtering_all_ILP2(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, const int R, const int W, const int H);

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// VECTORIZED VERSION (using AVX2)
void sure_all_VEC(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H);
void filtering_basic_VEC(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);
void filtering_basic_f3_VEC(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);   // Unrolled for specific f=3 and k_f = Inf
void filtering_basic_f1_VEC(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);   // Unrolled for specific f=1 and k_f = Inf
void feature_prefiltering_VEC(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_VEC(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_FIRST_VEC(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);  
void candidate_filtering_SECOND_VEC(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_THIRD_VEC(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_all_VEC(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int W, int H);

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// CLEAN BLOCKING ATTEMPT

void candidate_filtering_all_VEC_BLK(buffer output_r, 
                                         buffer output_g, 
                                         buffer output_b, 
                                         buffer color, 
                                         buffer color_var, 
                                         buffer features, 
                                         buffer features_var, 
                                         scalar* gradients, 
                                         scalar* features_weights_r,
                                         scalar* features_weights_b,
                                         scalar* temp,
                                         scalar* temp2_r, 
                                         scalar* temp2_g,
                                         Flt_parameters* p,
                                         int B_TYPE,
                                         int X0,
                                         int Y0, 
                                         int B_SIZE,
                                         int W, 
                                         int H);


void candidate_filtering_all_VEC_BLK_PREP(buffer output_r, 
                                         buffer output_g, 
                                         buffer output_b, 
                                         buffer color, 
                                         buffer color_var, 
                                         buffer features, 
                                         buffer features_var, 
                                         scalar* gradients, 
                                         scalar* features_weights_r,
                                         scalar* features_weights_b,
                                         scalar* temp,
                                         scalar* temp2_r, 
                                         scalar* temp2_g,
                                         Flt_parameters* p, 
                                         int W, 
                                         int H);


void candidate_filtering_all_VEC_BLK_noprec(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int X0, int Y0, int B_TYPE, int B_SIZE);


// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// BLOCKED VERSION (using AVX2)
void sure_all_BLK(buffer sure, buffer c, buffer c_var, buffer cand_r, buffer cand_g, buffer cand_b, int W, int H);
void filtering_basic_BLK(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);
void filtering_basic_f3_BLK(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);   // Unrolled for specific f=3 and k_f = Inf
void filtering_basic_f1_BLK(buffer output, buffer input, buffer c, buffer c_var, Flt_parameters p, int W, int H);   // Unrolled for specific f=1 and k_f = Inf
void feature_prefiltering_BLK(buffer output, buffer output_var, buffer features, buffer features_var, Flt_parameters p, int X0, int Y0, int W, int H);
void candidate_filtering_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_FIRST_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);  
void candidate_filtering_SECOND_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_THIRD_BLK(buffer output, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters p, int W, int H);
void candidate_filtering_all_BLK(buffer output_r, buffer output_g, buffer output_b, buffer color, buffer color_var, buffer features, buffer features_var, Flt_parameters* p, int X0, int Y0, int W, int H);


#endif //FLT_2_H